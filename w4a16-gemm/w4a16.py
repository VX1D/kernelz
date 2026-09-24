"""Fused int4-weight x bf16-activation GEMM for RDNA3 (gfx11) GPUs under ROCm PyTorch."""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch

GROUP_SIZE = 128
BLOCK_N = 128
_LIB_PATH = Path(__file__).resolve().parent / "libw4a16.so"
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _LIB_PATH.is_file():
            raise FileNotFoundError(f"{_LIB_PATH} is missing; run `python build.py` first")
        lib = ctypes.CDLL(str(_LIB_PATH))
        lib.w4a16_gemm.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        lib.w4a16_gemm.restype = ctypes.c_int
        _lib = lib
    return _lib


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an [N, K] weight to symmetric int4 with one fp16 scale per 128 columns."""
    n, k = weight.shape
    if k % GROUP_SIZE:
        raise ValueError(f"K must be a multiple of {GROUP_SIZE}")
    grouped = weight.float().reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    scales = grouped.abs().amax(-1).clamp_min(1e-8) / 7.0
    q = torch.round(grouped / scales[..., None]).clamp(-8, 7).to(torch.int8)
    u = (q + 8).to(torch.uint8).reshape(n, k)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous(), scales.to(torch.float16).contiguous()


def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reference bf16 weights, identical to what the kernel stages in LDS."""
    lo = (packed & 15).to(torch.int32) - 8
    hi = (packed >> 4).to(torch.int32) - 8
    q = torch.stack((lo, hi), -1).reshape(packed.shape[0], -1, GROUP_SIZE).float()
    return (q * scales.to(torch.bfloat16).float()[..., None]).to(torch.bfloat16).reshape(packed.shape[0], -1)


def supports(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype is torch.bfloat16 and x.dim() == 2 and x.shape[0] > 0
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
            and x.shape[1] % GROUP_SIZE == 0 and packed.dtype is torch.uint8 and packed.is_contiguous()
            and packed.shape == (packed.shape[0], x.shape[1] // 2) and packed.shape[0] % BLOCK_N == 0
            and scales.dtype is torch.float16 and scales.is_contiguous()
            and scales.shape == (packed.shape[0], x.shape[1] // GROUP_SIZE))


def linear(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """x[..., K] (bf16) @ dequantize(packed, scales)^T -> [..., N] (bf16). Forward only."""
    flat = x.reshape(-1, x.shape[-1])
    if not supports(flat, packed, scales):
        raise ValueError("unsupported input: need bf16 CUDA x with K % 128 == 0, N % 128 == 0, row stride % 8 == 0")
    out = torch.empty((flat.shape[0], packed.shape[0]), dtype=torch.bfloat16, device=x.device)
    rc = _load().w4a16_gemm(flat.data_ptr(), packed.data_ptr(), scales.data_ptr(), out.data_ptr(),
                            flat.shape[0], packed.shape[0], flat.shape[1], flat.stride(0), out.stride(0),
                            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream))
    if rc:
        raise RuntimeError(f"w4a16_gemm failed with code {rc}")
    return out.reshape(*x.shape[:-1], packed.shape[0])

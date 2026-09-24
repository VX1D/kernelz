"""Skinny Q4_0 x int8 GEMM for small batches (1-16 tokens) on RDNA3 (gfx11) GPUs under ROCm PyTorch."""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch

QK = 32
BLOCK_BYTES = 18
_LIB_PATH = Path(__file__).resolve().parent / "libq4_0_skinny.so"
_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _LIB_PATH.is_file():
            raise FileNotFoundError(f"{_LIB_PATH} is missing; run `python build.py` first")
        lib = ctypes.CDLL(str(_LIB_PATH))
        lib.q4_0_skinny_quantize.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.q4_0_skinny_gemm.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.q4_0_skinny_quantize.restype = lib.q4_0_skinny_gemm.restype = ctypes.c_int
        _lib = lib
    return _lib


def _stream(device):
    return ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)


def quantize_q4_0(weight: torch.Tensor) -> torch.Tensor:
    """[N, K] float -> ggml Q4_0 blocks as uint8 [N, K/32*18] (d = max/-8, q = round(x/d) + 8)."""
    n, k = weight.shape
    if k % QK:
        raise ValueError("K must be a multiple of 32")
    g = weight.float().reshape(n, k // QK, QK)
    idx = g.abs().argmax(-1, keepdim=True)
    d = g.gather(-1, idx).squeeze(-1) / -8.0
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.floor(g * inv[..., None] + 8.5), 0, 15).to(torch.uint8)
    qs = q[..., :16] | (q[..., 16:] << 4)
    dh = d.to(torch.float16).contiguous().view(torch.uint8).reshape(n, k // QK, 2)
    return torch.cat([dh, qs], dim=-1).reshape(n, -1).contiguous()


def dequantize_q4_0(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Reference float32 [N, K] weights from Q4_0 blocks."""
    b = packed.reshape(packed.shape[0], k // QK, BLOCK_BYTES)
    d = b[..., :2].contiguous().view(torch.float16).float()
    qs = b[..., 2:]
    q = torch.cat([qs & 15, qs >> 4], dim=-1).float() - 8.0
    return (q * d).reshape(packed.shape[0], k)


def quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [J, K] on the GPU -> (int8 [J, K], fp32 scales [J, K/32])."""
    j, k = x.shape
    if x.dtype is not torch.bfloat16 or x.stride(1) != 1 or k % QK:
        raise ValueError("need bf16 [J, K] with K % 32 == 0 and contiguous rows")
    aq = torch.empty((j, k), dtype=torch.int8, device=x.device)
    ad = torch.empty((j, k // QK), dtype=torch.float32, device=x.device)
    rc = _load().q4_0_skinny_quantize(x.data_ptr(), aq.data_ptr(), ad.data_ptr(), j, k, x.stride(0), _stream(x.device))
    if rc:
        raise RuntimeError(f"q4_0_skinny_quantize failed with code {rc}")
    return aq, ad


def gemm(packed: torch.Tensor, aq: torch.Tensor, ad: torch.Tensor) -> torch.Tensor:
    """Q4_0 weights [N, K/32*18] x quantized activations -> fp32 [J, N]."""
    j, k = aq.shape
    n = packed.shape[0]
    out = torch.empty((j, n), dtype=torch.float32, device=aq.device)
    rc = _load().q4_0_skinny_gemm(packed.data_ptr(), aq.data_ptr(), ad.data_ptr(), out.data_ptr(), n, k, j, _stream(aq.device))
    if rc:
        raise RuntimeError(f"q4_0_skinny_gemm failed with code {rc} (J must be 1..16, N/K must fit a tile config)")
    return out


def linear(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """bf16 x [J, K] @ dequant(packed)^T -> fp32 [J, N]."""
    return gemm(packed, *quantize_act(x))

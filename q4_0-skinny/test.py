"""Correctness against an exact float64 reference and weight bandwidth against bf16 torch.nn.functional.linear."""
import time

import torch

import q4_0_skinny as qs

SHAPES = [(3840, 4096), (3840, 2048), (4096, 3840), (3840, 15360), (15360, 3840)]  # (K, N)
TOKENS = [1, 4, 8, 16]
CACHE_BYTES = 256 << 20  # rotate weight copies so every call streams from VRAM, not the 64 MiB Infinity Cache


def timed(fns, iters=20):
    for f in fns:
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(iters):
        fns[i % len(fns)]()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    torch.manual_seed(0)
    print(f"{'K':>6} {'N':>6} {'J':>3} {'rel err':>9} {'skinny us':>10} {'GB/s':>6} {'bf16 us':>8} {'GB/s':>6} {'speedup':>8}")
    for k, n in SHAPES:
        packed = qs.quantize_q4_0(torch.randn(n, k, device="cuda") * 0.02)
        w = qs.dequantize_q4_0(packed, k)
        wb = w.to(torch.bfloat16)
        copies = max(1, CACHE_BYTES // (wb.numel() * 2))
        packs = [packed.clone() for _ in range(copies)]
        wbs = [wb.clone() for _ in range(copies)]
        for j in TOKENS:
            x = torch.randn(j, k, device="cuda").to(torch.bfloat16)
            aq, ad = qs.quantize_act(x)
            ref = ((aq.double().reshape(j, -1, 32) * ad.double()[..., None]).reshape(j, k)) @ w.double().t()
            err = ((qs.gemm(packed, aq, ad).double() - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-5, (k, n, j, err)
            t_q = timed([lambda p=p: qs.gemm(p, aq, ad) for p in packs])
            t_b = timed([lambda m=m: torch.nn.functional.linear(x, m) for m in wbs])
            print(f"{k:>6} {n:>6} {j:>3} {err:9.1e} {t_q * 1e6:10.1f} {packed.numel() / t_q / 1e9:6.0f} "
                  f"{t_b * 1e6:8.1f} {wb.numel() * 2 / t_b / 1e9:6.0f} {t_b / t_q:7.2f}x")
    print("PASS")


if __name__ == "__main__":
    main()

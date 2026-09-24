"""Correctness against an fp32 reference; throughput against torch.nn.functional.linear on pre-dequantized bf16 weights."""
import time

import torch

import w4a16

SHAPES = [(8192, 3840, 15360), (8192, 15360, 3840), (8192, 3840, 4096), (8192, 4096, 3840), (8192, 3840, 2048),
          (777, 3840, 4096), (1, 4096, 1024)]


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    torch.manual_seed(0)
    print(f"{'M':>6} {'K':>6} {'N':>6} {'err':>9} {'blas_err':>9} {'w4a16 TF':>9} {'bf16 blas TF':>12}")
    for m, k, n in SHAPES:
        packed, scales = w4a16.quantize(torch.randn(n, k, device="cuda") * 0.02)
        weight = w4a16.dequantize(packed, scales)
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        ref = x.float() @ weight.float().t()
        scale = ref.abs().max()
        err = ((w4a16.linear(x, packed, scales).float() - ref).abs().max() / scale).item()
        blas_err = ((torch.nn.functional.linear(x, weight).float() - ref).abs().max() / scale).item()
        assert err <= 2 * blas_err + 1e-6, (m, k, n, err, blas_err)
        flops = 2 * m * k * n
        t_mine = bench(lambda: w4a16.linear(x, packed, scales))
        t_blas = bench(lambda: torch.nn.functional.linear(x, weight))
        print(f"{m:>6} {k:>6} {n:>6} {err:9.2e} {blas_err:9.2e} {flops / t_mine / 1e12:9.1f} {flops / t_blas / 1e12:12.1f}")
    print("PASS")


if __name__ == "__main__":
    main()

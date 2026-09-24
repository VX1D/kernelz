# w4a16-gemm (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧

`out = x @ dequant(w).T` for RDNA3: int4 weights, bf16 activations, fp32 accumulation, bf16 output.
The weights are dequantized straight into LDS inside the GEMM, so there is no separate dequant
kernel and no bf16 copy of the weights in VRAM.

## Weight format

- Symmetric int4, `q` in `[-8, 7]`, stored as `q + 8` nibbles. Low nibble = even column.
- One fp16 scale per 128 columns (`packed`: `[N, K/2]` uint8, `scales`: `[N, K/128]` fp16).
- The kernel builds each weight as `bf16(q * float(bf16(scale)))` with round-to-nearest-even,
  bit-identical to `w4a16.dequantize`. Only the fp32 accumulation order differs from a vendor GEMM.

`w4a16.quantize(weight)` produces this format from any `[N, K]` float tensor.

## Usage

```sh
python build.py          # hipcc, gfx1100/gfx1101/gfx1102 by default
python test.py           # correctness + TFLOPS next to hipBLASLt
```

```python
import torch, w4a16

packed, scales = w4a16.quantize(weight)          # weight: [N, K]
y = w4a16.linear(x, packed, scales)              # x: [..., K] bf16 on the GPU -> [..., N] bf16
```

## Numbers

RX 7800 XT (gfx1101), ROCm 7.14, PyTorch 2.11. Error is the max relative error against an fp32
reference, hipBLASLt runs on the already dequantized bf16 weights.

| M | K | N | error w4a16 | error hipBLASLt | w4a16 TFLOPS | hipBLASLt TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 3840 | 15360 | 2.15e-3 | 2.15e-3 | **50.7** | 44.1 |
| 8192 | 15360 | 3840 | 2.23e-3 | 2.22e-3 | **49.8** | 45.3 |
| 8192 | 3840 | 4096 | 2.21e-3 | 2.21e-3 | **48.5** | 44.1 |
| 8192 | 4096 | 3840 | 2.29e-3 | 2.29e-3 | 44.9 | 44.6 |
| 8192 | 3840 | 2048 | 2.17e-3 | 2.17e-3 | **46.9** | 42.9 |
| 777 | 3840 | 4096 | 2.37e-3 | 2.37e-3 | 34.2 | 37.3 |
| 1 | 4096 | 1024 | 2.84e-3 | 2.84e-3 | 0.1 | 0.2 |

Same accuracy as hipBLASLt, faster for big M, and the weights take 4× less memory.
Sustained bf16 WMMA peak on this card is about 64 TFLOPS, so the big shapes sit at ~75-80% of it.

## How it works

- 256×128 output tile per block, 8 waves, each wave owns a 64×64 tile (4×4 WMMA 16×16×16).
- K steps of 32, double-buffered LDS, padded rows so fragment loads are bank-conflict free.
- The next stage is dequantized between the two WMMA halves of the current one, so VALU work
  interleaves with WMMA instead of stalling after it.

## Limits

- Forward only, no backward.
- `N % 128 == 0`, `K % 128 == 0`, bf16 input with row stride divisible by 8.
- Built for prefill / big batches. With M below a few hundred rows the 256-row tile wastes most of
  the work, and decode (M = 1) is way slower than a GEMV. Use something else there (◞‸◟)
- RDNA3 only (`__builtin_amdgcn_wmma_*_w32`), tested on gfx1101.

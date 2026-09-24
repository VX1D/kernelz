# q4_0-skinny (•̀ᴗ•́)و

Q4_0 weights × int8 activations for small batches: 1 to 16 tokens, the decode / speculative
verify regime where a normal GEMM tile wastes most of the GPU. `out[J, N] = act[J, K] @ dequant(w)^T`
in fp32.

Weights stream from VRAM straight into WMMA int8 operands (no LDS staging), each block owns 16 weight
rows × 16 tokens and its waves split K, then reduce in LDS. The grid is sized to keep all 60 CUs busy
even when N is only 2048.

## Formats

- **Weights:** the ggml Q4_0 block, 18 bytes per 32 values: fp16 scale + 16 bytes, low nibbles are
  values 0..15, high nibbles 16..31, stored `q + 8`. Straight from a GGUF, no repacking.
- **Activations:** int8 per value + fp32 scale per 32 values (`d = amax / 127`, `q = round(x / d)`,
  same values as ggml's q8_1). `quantize_act` does it on the GPU from bf16.

## Usage

```sh
python build.py          # hipcc, gfx1100/gfx1101/gfx1102 by default
python test.py           # correctness + bandwidth next to bf16 F.linear
```

```python
import q4_0_skinny as qs

packed = qs.quantize_q4_0(weight)       # [N, K] float -> [N, K/32*18] uint8, or load Q4_0 from a GGUF
y = qs.linear(x, packed)                # x: [J, K] bf16 on the GPU, J <= 16 -> [J, N] fp32
```

## Numbers

RX 7800 XT (gfx1101, ~624 GB/s peak), ROCm 7.14, PyTorch 2.11. Weight copies are rotated so every
call streams from VRAM instead of the 64 MiB Infinity Cache. Error is against an exact float64
reference of the same quantized math. The bf16 column is `F.linear` on the dequantized weights,
which reads 3.6× more bytes, so GB/s is the fair comparison and the speedup is what you get in practice.

| K | N | J | rel err | skinny µs | GB/s | bf16 µs | GB/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3840 | 4096 | 1 | 1.3e-7 | 24.0 | 369 | 100.1 | 314 | 4.17× |
| 3840 | 4096 | 16 | 1.4e-7 | 24.4 | 363 | 102.7 | 306 | 4.21× |
| 3840 | 2048 | 1 | 1.3e-7 | 15.6 | 283 | 66.0 | 238 | 4.23× |
| 3840 | 2048 | 16 | 1.2e-7 | 16.3 | 272 | 66.8 | 235 | 4.10× |
| 4096 | 3840 | 1 | 1.7e-7 | 24.6 | 360 | 181.3 | 173 | 7.37× |
| 4096 | 3840 | 16 | 1.9e-7 | 27.2 | 325 | 166.8 | 189 | 6.13× |
| 3840 | 15360 | 1 | 1.4e-7 | 73.9 | 449 | 347.0 | 340 | 4.70× |
| 3840 | 15360 | 16 | 1.7e-7 | 77.7 | 427 | 348.5 | 338 | 4.49× |
| 15360 | 3840 | 1 | 2.3e-7 | 64.9 | 511 | 390.6 | 302 | 6.02× |
| 15360 | 3840 | 16 | 1.8e-7 | 64.5 | 514 | 389.7 | 303 | 6.04× |

J = 1 and J = 16 cost the same: the kernel is bandwidth-bound, so batching up to 16 tokens is free.
The big matrices get to ~80% of peak bandwidth, the small ones are latency-bound (15 µs is not a lot
of time to fill 60 CUs).

## Limits

- 1 ≤ J ≤ 16, K % 32 == 0, and N must fit a tile config (multiple of 16 or 32 depending on shape).
- The tile table was tuned on 3840/4096/15360-wide matrices; other shapes work but aren't tuned.
- Activations are quantized to int8, so this is not bit-comparable to a bf16 GEMM (it matches
  llama.cpp's Q4_0 × q8_1 math instead).
- RDNA3 only (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`), tested on gfx1101 (◞‸◟)

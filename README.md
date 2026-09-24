# kernelz ٩(◕‿◕)۶

One folder per kernel. Each one has the source, a tiny Python binding, a build script and a test
that checks correctness and flexes the numbers next to the vendor library.

| Kernel | What it does | Hardware | Status |
|---|---|---|---|
| [w4a16-gemm](w4a16-gemm) | int4 weights × bf16 activations GEMM, dequant happens inside the kernel | RDNA3 (gfx11) | Works on my machine™ |
| [q4_0-skinny](q4_0-skinny) | Q4_0 × int8 GEMM for 1 to 16 tokens, 4-7× faster than bf16 `F.linear` | RDNA3 (gfx11) | Works on my machine™ |

## Rules of the house

- One folder = one kernel: `<name>.hip`, `<name>.py`, `build.py`, `test.py`, `README.md`.
- Every test compares against a higher-precision reference, no vibes-based correctness.
- Numbers are measured on the GPU written next to them. Your card, your numbers.

More kernels coming whenever something is too slow again (✿◕‿◕)

## License

MIT, see [LICENSE](LICENSE).

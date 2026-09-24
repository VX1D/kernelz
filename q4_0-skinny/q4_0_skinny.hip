// Skinny Q4_0 x int8 GEMM for small batches (J <= 16 tokens) on RDNA3 (gfx11) WMMA.
// out[J, N] (fp32) = act[J, K] @ dequant(w)[N, K]^T, with act quantized per 32 values to int8 + fp32 scale.
// w uses the ggml Q4_0 block: fp16 scale + 16 bytes, low nibbles = values 0..15, high = 16..31, stored q + 8.
// One block owns 16*R weight rows x 16 tokens; its NW waves split K and reduce in LDS.
// Weights stream from global straight into WMMA operands, no LDS staging.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

constexpr int QK = 32;

struct __attribute__((packed)) block_q4_0 {
    _Float16 d;
    uint8_t qs[QK / 2];
};
static_assert(sizeof(block_q4_0) == 18, "ggml Q4_0 block is 18 bytes");

// Nibbles 0..15 -> signed int8 -8..7, four at a time without cross-byte borrow.
__device__ __forceinline__ int nib_to_i8(const int v) {
    return ((v | 0x80808080) - 0x08080808) ^ 0x80808080;
}

__device__ __forceinline__ int load_i32_unaligned(const uint8_t * p) {
    int v;
    __builtin_memcpy(&v, p, sizeof(int));
    return v;
}

template <int R, int NW, int CB>
__global__ void __launch_bounds__(32 * NW, 1)
q4_0_skinny_kernel(const block_q4_0 * __restrict__ w, const int8_t * __restrict__ aq,
                   const float * __restrict__ ad, float * __restrict__ out, int N, int K, int J) {
    constexpr int ROWS = 16 * R;
    const int nblocks = K / QK;
    const int lane = threadIdx.x, wave = threadIdx.y;
    const int i16 = lane % 16;
    const int row0 = blockIdx.x * ROWS;
    // Padded token columns read a valid token; their results are never written.
    const int j_load = min(i16, J - 1);

    const block_q4_0 * wrow[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wrow[r] = w + (size_t) (row0 + 16 * r + i16) * nblocks;
    }
    const int8_t * arow = aq + (size_t) j_load * K;
    const float * drow = ad + (size_t) j_load * nblocks;

    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int l = 0; l < 8; ++l)
            acc[r][l] = 0.0f;

    // One chunk = CB consecutive Q4_0 blocks; two chunks are in flight per wave.
    struct chunk_t {
        int qa[R][CB][4];
        float da[R][CB];
        int qb[CB][8];
        float db[CB];
    };

    const auto load_chunk = [&](const int kb0, chunk_t & ch) {
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            const int kb = kb0 + c;
            const int * aqb = (const int *) (arow + kb * QK);
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                ch.qb[c][m] = aqb[m];
            }
            ch.db[c] = drow[kb];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const block_q4_0 * wb = wrow[r] + kb;
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    ch.qa[r][c][m] = load_i32_unaligned(wb->qs + 4 * m);
                }
                ch.da[r][c] = (float) wb->d;
            }
        }
    };

    const auto compute_chunk = [&](const chunk_t & ch) {
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            i32x4 b_lo, b_hi;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                b_lo[m] = ch.qb[c][m];
                b_hi[m] = ch.qb[c][4 + m];
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                i32x4 a_lo, a_hi;
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    a_lo[m] = nib_to_i8(ch.qa[r][c][m] & 0x0F0F0F0F);
                    a_hi[m] = nib_to_i8((ch.qa[r][c][m] >> 4) & 0x0F0F0F0F);
                }
                i32x8 d = {0, 0, 0, 0, 0, 0, 0, 0};
                d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_lo, true, b_lo, d, false);
                d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_hi, true, b_hi, d, false);
                // RDNA3 C layout: d[l] = D[weight row 2*l + lane/16][token lane%16].
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const float dw = __shfl(ch.da[r][c], 2 * l + lane / 16, 32);
                    acc[r][l] = fmaf((float) d[l], dw * ch.db[c], acc[r][l]);
                }
            }
        }
    };

    // Waves walk chunks of CB blocks interleaved across the block; nblocks % CB == 0.
    chunk_t cur, nxt;
    int kb0 = wave * CB;
    if (kb0 < nblocks) {
        load_chunk(kb0, cur);
    }
    for (; kb0 < nblocks; kb0 += NW * CB) {
        const int kb_next = kb0 + NW * CB;
        if (kb_next < nblocks) {
            load_chunk(kb_next, nxt);
        }
        compute_chunk(cur);
        cur = nxt;
    }

    __shared__ float red[NW > 1 ? NW - 1 : 1][R][8][32];
    if (wave > 0) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int l = 0; l < 8; ++l)
                red[wave - 1][r][l][lane] = acc[r][l];
    }
    __syncthreads();
    if (wave != 0) {
        return;
    }
#pragma unroll
    for (int wv = 0; wv < NW - 1; ++wv)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int l = 0; l < 8; ++l)
                acc[r][l] += red[wv][r][l][lane];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int row = row0 + 16 * r + 2 * l + lane / 16;
            if (i16 < J) {
                out[(size_t) i16 * N + row] = acc[r][l];
            }
        }
    }
}

// One thread per 32-value block: d = amax / 127, q = round(x / d), matching ggml's q8_1 values.
__global__ void quantize_act_kernel(const uint16_t * __restrict__ x, int8_t * __restrict__ aq,
                                    float * __restrict__ ad, int J, int K, int ldx) {
    const int nblocks = K / QK;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= J * nblocks) {
        return;
    }
    const int j = id / nblocks, kb = id % nblocks;
    const uint16_t * src = x + (size_t) j * ldx + kb * QK;
    float v[QK];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < QK; ++i) {
        v[i] = __uint_as_float(((uint32_t) src[i]) << 16);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float d = amax / 127.0f;
    int8_t * dst = aq + (size_t) j * K + kb * QK;
#pragma unroll
    for (int i = 0; i < QK; ++i) {
        dst[i] = amax == 0.0f ? 0 : (int8_t) roundf(v[i] / d);
    }
    ad[(size_t) j * nblocks + kb] = d;
}

struct cfg_t { int R, NW, CB; };

// Best of an R x NW x CB sweep at J = 16 on RX 7800 XT for 3840/4096/15360-wide matrices.
cfg_t pick_cfg(int N, int K) {
    if (K >= 8192 || N >= 8192) return {1, 8, 4};
    if (N <= 2048) return {1, 8, 2};
    if (K > N) return {2, 4, 2};
    return {1, 4, 2};
}

template <int R, int NW, int CB>
void launch(const void * w, const int8_t * aq, const float * ad, float * out, int N, int K, int J, hipStream_t s) {
    hipLaunchKernelGGL((q4_0_skinny_kernel<R, NW, CB>), dim3(N / (16 * R)), dim3(32, NW), 0, s,
                       (const block_q4_0 *) w, aq, ad, out, N, K, J);
}

} // namespace

// Quantizes bf16 activations [J, K] (row stride ldx) into aq [J, K] int8 and ad [J, K/32] fp32.
extern "C" int q4_0_skinny_quantize(const void * x, void * aq, void * ad, int J, int K, int ldx, void * stream) {
    if (J < 1 || K % QK != 0) {
        return 1;
    }
    const int n = J * (K / QK);
    hipLaunchKernelGGL(quantize_act_kernel, dim3((n + 255) / 256), dim3(256), 0, (hipStream_t) stream,
                       (const uint16_t *) x, (int8_t *) aq, (float *) ad, J, K, ldx);
    return hipGetLastError() == hipSuccess ? 0 : 2;
}

// out[J, N] fp32. Returns 0 on success, 1 for unsupported shapes, 2 for a launch error.
extern "C" int q4_0_skinny_gemm(const void * w, const void * aq, const void * ad, void * out,
                                int N, int K, int J, void * stream) {
    if (J < 1 || J > 16 || K % QK != 0) {
        return 1;
    }
    const cfg_t c = pick_cfg(N, K);
    if ((K / QK) % c.CB != 0 || N % (16 * c.R) != 0) {
        return 1;
    }
    hipStream_t s = (hipStream_t) stream;
    const int8_t * q = (const int8_t *) aq;
    const float * d = (const float *) ad;
    float * o = (float *) out;
    if (c.R == 1 && c.NW == 8 && c.CB == 4) launch<1, 8, 4>(w, q, d, o, N, K, J, s);
    else if (c.R == 1 && c.NW == 8 && c.CB == 2) launch<1, 8, 2>(w, q, d, o, N, K, J, s);
    else if (c.R == 2 && c.NW == 4 && c.CB == 2) launch<2, 4, 2>(w, q, d, o, N, K, J, s);
    else launch<1, 4, 2>(w, q, d, o, N, K, J, s);
    return hipGetLastError() == hipSuccess ? 0 : 2;
}

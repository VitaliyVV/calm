/**
 * calm_quant.c — Calm Quantized Tensor Kernels
 *
 * NEON-optimized matmul for ARM aarch64 (Snapdragon 8+ Gen 1).
 * Supports BQ1_0 (binary 1-bit), TQ1_0 (ternary 1.58-bit),
 * Q8_0, Q4_0, and F32 formats.
 *
 * Сборка: clang -O3 -std=c11 -march=native -DCT_NEON -c calm_quant.c
 */

#include "calm_quant.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════
 * FP16 <-> FP32 (table-based for accuracy)
 * ═══════════════════════════════════════════════════════════════ */

static float ct_fp16_table[1 << 16];
static bool ct_fp16_table_init = false;

static void ct_fp16_init(void) {
    if (ct_fp16_table_init) return;
    /* Verify the table isn't uninitialized by checking a known value first */
    for (int i = 0; i < (1 << 16); i++) {
        uint16_t h = (uint16_t)i;
        /* Correct FP16 → FP32 bit conversion */
        uint32_t sign   = ((uint32_t)h & 0x8000) << 16;
        uint32_t exp_f16 = ((uint32_t)h >> 10) & 0x1F;
        uint32_t mant   = (uint32_t)h & 0x03FF;
        uint32_t r;
        if (exp_f16 == 0) {
            if (mant == 0) {
                r = sign;                           /* ±0 */
            } else {
                /* Subnormal: normalize by shifting mantissa */
                int shift = 10;
                uint32_t m = mant;
                while ((m & 0x0400) == 0) { m <<= 1; shift--; }
                m &= 0x03FF;
                uint32_t exp_f32 = (uint32_t)(127 - 14 - shift);
                r = sign | (exp_f32 << 23) | (m << 13);
            }
        } else if (exp_f16 == 31) {
            r = sign | 0x7F800000U | (mant << 13);  /* inf/nan */
        } else {
            uint32_t exp_f32 = exp_f16 + 127 - 15;
            r = sign | (exp_f32 << 23) | (mant << 13);
        }
        memcpy(&ct_fp16_table[i], &r, sizeof(r));
    }
    ct_fp16_table_init = true;
}

static inline float fp16_to_f32(uint16_t h) {
    float v = ct_fp16_table[h];
    if (h == 0x3C00 && v < 0.5f) {
        fprintf(stderr, "[fp16] WARN: table[0x3C00]=%f (expected 1.0) table_init=%d\n",
                v, ct_fp16_table_init);
    }
    return v;
}

/* ═══════════════════════════════════════════════════════════════
 * PTQ Calibration: MSE-optimal scale/threshold for 1-bit formats
 * ═══════════════════════════════════════════════════════════════ */

/* Comparator for float sort (ascending) */
static int cmp_float_asc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

float ct_calibrate_binary(const float* x, int n, float* out_mse) {
    /* BQ1_0: quant(w) = α·sign(w)
     * MSE = Σ(|wᵢ| - α)²
     * dMSE/dα = 2·Σ(α - |wᵢ|) = 2(n·α - Σ|wᵢ|) = 0 → α = mean(|w|)
     * So mean(|w|) is already MSE-optimal.
     */
    if (n <= 0) { if (out_mse) *out_mse = 0.0f; return 1.0f; }
    float sum_abs = 0.0f;
    for (int i = 0; i < n; i++) sum_abs += fabsf(x[i]);
    float alpha = sum_abs / (float)n;
    if (alpha < 1e-10f) alpha = 1.0f;
    if (out_mse) {
        float mse = 0.0f;
        for (int i = 0; i < n; i++) {
            float err = fabsf(x[i]) - alpha;
            mse += err * err;
        }
        *out_mse = mse / (float)n;
    }
    return alpha;
}

float ct_calibrate_ternary(const float* x, int n, float* out_mse) {
    /* TQ1_0: quant(w, α) = 0 if |w| < α/2, else α·sign(w)
     *
     * Find α that minimizes MSE by sweeping split points:
     * 1. Sort |w| ascending
     * 2. For split k (first k weights → 0, remaining n-k → ±α):
     *    α = mean(|w|_remaining)
     *    must satisfy: |w_k| < α/2 ≤ |w_{k+1}|  (consistency)
     * 3. Pick split with lowest MSE
     *
     * Uses stack allocation for n ≤ CT_TQ1_0_BLOCK_SIZE (256),
     * falls back to heap for larger arrays.
     */
    if (n <= 0) { if (out_mse) *out_mse = 0.0f; return 1.0f; }

    /* Allocate sorted abs values — stack for small n, heap for large */
    float abs_vals_stack[CT_TQ1_0_BLOCK_SIZE];
    double prefix_stack[CT_TQ1_0_BLOCK_SIZE + 1];
    float* abs_vals = (n <= CT_TQ1_0_BLOCK_SIZE) ? abs_vals_stack
                     : (float*)malloc((size_t)n * sizeof(float));
    double* prefix = (n <= CT_TQ1_0_BLOCK_SIZE) ? prefix_stack
                    : (double*)malloc((size_t)(n + 1) * sizeof(double));

    if ((size_t)n > CT_TQ1_0_BLOCK_SIZE && (!abs_vals || !prefix)) {
        free(abs_vals); free(prefix);
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += fabsf(x[i]);
        float d = sum / (float)n;
        if (d < 1e-10f) d = 1.0f;
        if (out_mse) *out_mse = 0.0f;
        return d;
    }

    for (int i = 0; i < n; i++) abs_vals[i] = fabsf(x[i]);
    qsort(abs_vals, (size_t)n, sizeof(float), cmp_float_asc);

    prefix[0] = 0.0;
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + (double)abs_vals[i];

    float best_alpha = abs_vals[n - 1];
    float best_mse = 1e30f;

    for (int k = 0; k <= n - 2; k++) {
        int nz = n - k;
        double sum_nz = prefix[n] - prefix[k];
        float alpha = (float)(sum_nz / (double)nz);
        if (alpha < 1e-10f) continue;
        float half_alpha = alpha * 0.5f;

        if (k > 0 && abs_vals[k - 1] >= half_alpha) continue;
        if (abs_vals[k] < half_alpha) continue;

        double mse = 0.0;
        for (int i = 0; i < k; i++)
            mse += (double)abs_vals[i] * (double)abs_vals[i];
        for (int i = k; i < n; i++) {
            double err = (double)abs_vals[i] - (double)alpha;
            mse += err * err;
        }
        float mse_f = (float)(mse / (double)n);
        if (mse_f < best_mse) { best_mse = mse_f; best_alpha = alpha; }
    }

    /* Fallback if no consistent split found */
    if (best_mse > 1e29f) {
        double sum_all = prefix[n];
        best_alpha = (float)(sum_all / (double)n);
        if (best_alpha < 1e-10f) best_alpha = 1.0f;
        if (out_mse) {
            double mse = 0.0;
            for (int i = 0; i < n; i++) {
                double err = (double)abs_vals[i] - (double)best_alpha;
                mse += err * err;
            }
            best_mse = (float)(mse / (double)n);
        }
    }

    if ((size_t)n > CT_TQ1_0_BLOCK_SIZE) {
        free(abs_vals);
        free(prefix);
    }
    if (out_mse) *out_mse = best_mse;
    return best_alpha;
}

#if defined(__ARM_NEON)
#include <arm_neon.h>

/* ═══════════════════════════════════════════════════════════════
 * NEON: Dequantization
 * ═══════════════════════════════════════════════════════════════ */

void ct_dequant_q8_0(const ct_block_q8_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 32 ? count : 32;
    for (int i = 0; i < n; i++)
        out[i] = block->qs[i] * d;
    for (int i = n; i < count; i++)
        out[i] = 0;
}

void ct_dequant_q4_0(const ct_block_q4_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 32 ? count : 32;
    for (int i = 0; i < n; i++) {
        int nib = (block->qs[i >> 1] >> ((i & 1) << 2)) & 0xF;
        out[i] = ((float)nib - 8.0f) * d;
    }
    for (int i = n; i < count; i++)
        out[i] = 0;
}

void ct_dequant_bq1_0(const ct_block_bq1_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 128 ? count : 128;
    for (int i = 0; i < n; i++) {
        int byte_idx = i >> 6;       /* i / 64 */
        int bit_idx = i & 0x3F;      /* i % 64 */
        out[i] = ((block->bits[byte_idx] >> bit_idx) & 1) ? d : -d;
    }
    for (int i = n; i < count; i++)
        out[i] = 0;
}

/* TQ1_0 dequant: base-3 unpacking */
/* 5 ternary values per byte (3^5 = 243 < 256) */
static const int tq1_pow3[5] = {1, 3, 9, 27, 81};
static inline void tq1_unpack_byte(uint8_t byte, int vals[5]) {
    int tmp = byte;
    for (int i = 0; i < 5; i++) {
        vals[i] = (tmp % 3) - 1;  /* 0→-1, 1→0, 2→+1 */
        tmp /= 3;
    }
}

void ct_dequant_tq1_0(const ct_block_tq1_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 256 ? count : 256;
    int idx = 0;
    /* First 240 values from qs (48 bytes × 5 values each) */
    for (int i = 0; i < 48 && idx < n; i++) {
        int vals[5];
        tq1_unpack_byte(block->qs[i], vals);
        for (int j = 0; j < 5 && idx < n; j++)
            out[idx++] = (float)vals[j] * d;
    }
    /* Last 16 values from qh (4 bytes × 4 values each, 2-bit packed) */
    for (int i = 0; i < 4 && idx < n; i++) {
        for (int j = 0; j < 4 && idx < n; j++) {
            int v = (block->qh[i] >> (j * 2)) & 3;
            out[idx++] = (float)(v - 1) * d;  /* 0→-1, 1→0, 2→+1, 3→+1 */
        }
    }
    for (; idx < count; idx++)
        out[idx] = 0;
}

/* ═══════════════════════════════════════════════════════════════
 * NEON: Quantization
 * ═══════════════════════════════════════════════════════════════ */

void ct_quant_q8_0(const float* x, ct_block_q8_0* block, int count) {
    int n = count < 32 ? count : 32;
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0) {
        block->d = ct_fp32_to_fp16(1.0f);
        memset(block->qs, 0, n);
        return;
    }
    float d = amax / 127.0f;
    float id = 1.0f / d;
    for (int i = 0; i < n; i++)
        block->qs[i] = (int8_t)lrintf(x[i] * id);
    block->d = ct_fp32_to_fp16(d);
    for (int i = n; i < 32; i++) block->qs[i] = 0;
}

void ct_quant_q4_0(const float* x, ct_block_q4_0* block, int count) {
    int n = count < 32 ? count : 32;
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0) {
        block->d = ct_fp32_to_fp16(1.0f);
        memset(block->qs, 0, 16);
        return;
    }
    float d = amax / 7.0f;
    float id = 1.0f / d;
    memset(block->qs, 0, 16);
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(x[i] * id) + 8;
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        if (i & 1)
            block->qs[i >> 1] |= (uint8_t)(q << 4);
        else
            block->qs[i >> 1] |= (uint8_t)q;
    }
    block->d = ct_fp32_to_fp16(d);
}

void ct_quant_bq1_0(const float* x, ct_block_bq1_0* block, int count) {
    int n = count < 128 ? count : 128;
    /* Compute scale = mean(|x|) */
    float sum_abs = 0.0f;
    for (int i = 0; i < n; i++)
        sum_abs += fabsf(x[i]);
    float d = sum_abs / (float)n;
    if (d < 1e-10f) d = 1.0f;

    block->bits[0] = 0;
    block->bits[1] = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] >= 0)
            block->bits[i >> 6] |= (uint64_t)1 << (i & 0x3F);
    }
    block->d = ct_fp32_to_fp16(d);
    /* Zero-pad remaining */
    if (n < 128) {
        for (int i = n; i < 128; i++) {
            /* sign doesn't matter for zero */
        }
    }
}

/* TQ1_0 quantize: 5 ternary values per byte (base-3)
 * Uses MSE-optimal calibrated scale per block via ct_calibrate_ternary.
 */
void ct_quant_tq1_0(const float* x, ct_block_tq1_0* block, int count) {
    int n = count < 256 ? count : 256;

    /* Compute MSE-optimal scale via calibration */
    float d = ct_calibrate_ternary(x, n, NULL);
    if (d < 1e-10f) d = 1.0f;
    float id = 1.0f / d;

    memset(block->qs, 0, sizeof(block->qs));
    memset(block->qh, 0, sizeof(block->qh));
    block->d = ct_fp32_to_fp16(d);

    int idx = 0;
    /* First 240 values → qs (5 per byte, base-3) */
    for (int i = 0; i < 48 && idx < n; i++) {
        int byte_val = 0;
        int mult = 1;
        for (int j = 0; j < 5 && idx < n; j++) {
            float v = x[idx] * id;
            int t;
            if (v > 0.5f) t = 2;        /* +1 */
            else if (v < -0.5f) t = 0;  /* -1 */
            else t = 1;                  /* 0 */
            byte_val += t * mult;
            mult *= 3;
            idx++;
        }
        block->qs[i] = (uint8_t)byte_val;
    }
    /* Last 16 values → qh (2-bit each, 4 per byte) */
    for (int i = 0; i < 4 && idx < n; i++) {
        uint8_t byte_val = 0;
        for (int j = 0; j < 4 && idx < n; j++) {
            float v = x[idx] * id;
            int t;
            if (v > 0.5f) t = 2;
            else if (v < -0.5f) t = 0;
            else t = 1;
            byte_val |= (uint8_t)t << (j * 2);
            idx++;
        }
        block->qh[i] = byte_val;
    }
}

/* TQ1_0 fast quantize: original mean(|w|) without PTQ calibration */
void ct_quant_tq1_0_fast(const float* x, ct_block_tq1_0* block, int count) {
    int n = count < 256 ? count : 256;
    float sum_abs = 0.0f;
    for (int i = 0; i < n; i++)
        sum_abs += fabsf(x[i]);
    float d = sum_abs / (float)n;
    if (d < 1e-10f) d = 1.0f;
    float id = 1.0f / d;

    memset(block->qs, 0, sizeof(block->qs));
    memset(block->qh, 0, sizeof(block->qh));
    block->d = ct_fp32_to_fp16(d);

    int idx = 0;
    for (int i = 0; i < 48 && idx < n; i++) {
        int byte_val = 0;
        int mult = 1;
        for (int j = 0; j < 5 && idx < n; j++) {
            float v = x[idx] * id;
            int t;
            if (v > 0.5f) t = 2;
            else if (v < -0.5f) t = 0;
            else t = 1;
            byte_val += t * mult;
            mult *= 3;
            idx++;
        }
        block->qs[i] = (uint8_t)byte_val;
    }
    for (int i = 0; i < 4 && idx < n; i++) {
        uint8_t byte_val = 0;
        for (int j = 0; j < 4 && idx < n; j++) {
            float v = x[idx] * id;
            int t;
            if (v > 0.5f) t = 2;
            else if (v < -0.5f) t = 0;
            else t = 1;
            byte_val |= (uint8_t)t << (j * 2);
            idx++;
        }
        block->qh[i] = byte_val;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TQ1_0 → Q8_0 dequant for NEON matmul reuse
 *
 * TQ1_0 block (256 elements): base-3 packed {−1,0,+1} × d
 * Q8_0 block (32 elements):   int8 × d
 *
 * Ternary values fit directly into int8, so we decode TQ1_0 →
 * int8 packs and set Q8_0 scale = TQ1_0 scale. No float needed.
 * ═══════════════════════════════════════════════════════════════ */

/* Decode TQ1_0 block into 8 × Q8_0 blocks (256 elements → 8×32).
 * out_q8 must point to at least 8 × CT_SIZEOF_Q8_0 bytes.
 * Returns the number of valid elements (256 or less for partial block).
 */
static int dequant_tq1_0_to_q8_0(const ct_block_tq1_0* tq,
                                   ct_block_q8_0* out_q8, int cols) {
    int n = cols < 256 ? cols : 256;
    int idx = 0;

    /* Decode all 256 ternary values into a temporary int8 array */
    int8_t tmp[256];
    memset(tmp, 0, sizeof(tmp));

    /* First 240 values from qs (48 bytes × 5 per byte, base-3) */
    for (int i = 0; i < 48 && idx < 256; i++) {
        int byte_val = tq->qs[i];
        for (int j = 0; j < 5 && idx < 256; j++) {
            int t = (byte_val % 3) - 1;  /* 0→-1, 1→0, 2→+1 */
            byte_val /= 3;
            tmp[idx++] = (int8_t)t;
        }
    }
    /* Last 16 values from qh (4 bytes × 4 per byte, 2-bit) */
    for (int i = 0; i < 4 && idx < 256; i++) {
        uint8_t byte_val = tq->qh[i];
        for (int j = 0; j < 4 && idx < 256; j++) {
            int v = (byte_val >> (j * 2)) & 3;
            tmp[idx++] = (int8_t)(v - 1);  /* 0→-1, 1→0, 2→+1, 3→+2(clamp) */
        }
    }

    /* Pack into Q8_0 blocks (32 elements each) */
    int nb = (n + 31) / 32;
    for (int b = 0; b < nb; b++) {
        out_q8[b].d = tq->d;  /* same scale as TQ1_0 */
        int base = b * 32;
        for (int i = 0; i < 32; i++) {
            int idx_q = base + i;
            out_q8[b].qs[i] = (idx_q < n) ? tmp[idx_q] : 0;
        }
    }
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * NEON: Q8_0 Matmul — y[O] = x[I] @ W^T
 *
 * GGUF stores weights as [O][I] (output rows × input columns).
 * For each output j:
 *   y[j] = Σ_i x[i] * d_j[i/32] * qs_j[i/32][i%32]
 * Uses vaddvq_f32 to reduce 4-vector to scalar.
 * ═══════════════════════════════════════════════════════════════ */

void ct_matmul_q8_0(float* y, const float* x,
                     const ct_block_q8_0* W, int I, int O) {
    int blk_per_I = (I + 31) / 32;

    for (int j = 0; j < O; j++) {
        const ct_block_q8_0* row = W + (int64_t)j * blk_per_I;
        float32x4_t vacc = vdupq_n_f32(0.0f);

        for (int b = 0; b < blk_per_I; b++) {
            float d = fp16_to_f32(row[b].d);
            float32x4_t dv = vdupq_n_f32(d);
            int i0 = b * 32;

            for (int g = 0; g < 4; g++) {
                int16x8_t w16 = vmovl_s8(vld1_s8(row[b].qs + g * 8));
                float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
                float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));

                int ii = i0 + g * 8;
                if (ii + 4 > I) break;  /* partial block remainder */
                float32x4_t x_lo = vld1q_f32(x + ii);
                float32x4_t x_hi = vld1q_f32(x + ii + 4);
                vacc = vfmaq_f32(vacc, x_lo, vmulq_f32(dv, wf_lo));
                vacc = vfmaq_f32(vacc, x_hi, vmulq_f32(dv, wf_hi));
            }
        }
        y[j] = vaddvq_f32(vacc);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * NEON: TQ1_0 Matmul via Q8_0 path
 *
 * Dequantizes TQ1_0 → Q8_0 blocks on-the-fly, then delegates to
 * the NEON-optimized ct_matmul_q8_0(). Reuses existing SIMD code.
 * ═══════════════════════════════════════════════════════════════ */

void ct_matmul_tq1_0(float* y, const float* x,
                      const ct_block_tq1_0* W, int I, int O) {
    int nb_per_row_tq = (I + 255) / 256;

    /* Temporary Q8_0 buffer for one row */
    int max_q8_blocks = (I + 31) / 32;
    ct_block_q8_0* q8_buf = (ct_block_q8_0*)malloc((size_t)max_q8_blocks * CT_SIZEOF_Q8_0);
    if (!q8_buf) {
        memset(y, 0, (size_t)O * sizeof(float));
        return;
    }

    for (int j = 0; j < O; j++) {
        const ct_block_tq1_0* tq_row = W + (int64_t)j * nb_per_row_tq;
        int q8_count = 0;

        /* Dequant all TQ1_0 blocks in this row → Q8_0 blocks */
        for (int b = 0; b < nb_per_row_tq; b++) {
            int cols_in_block = I - b * 256;
            if (cols_in_block <= 0) break;
            dequant_tq1_0_to_q8_0(&tq_row[b], &q8_buf[q8_count], cols_in_block);
            q8_count += (cols_in_block + 31) / 32;
        }

        /* Run NEON Q8_0 matmul for this output row */
        ct_matmul_q8_0(y + j, x, q8_buf, I, 1);
    }

    free(q8_buf);
}

/* ═══════════════════════════════════════════════════════════════
 * NEON: Q4_0 Matmul — y[O] = x[I] @ W^T (stored as [O][I])
 *
 * Nibbles unpacked to int8 [-8,7], then same dot-product path as Q8_0
 * ═══════════════════════════════════════════════════════════════ */

void ct_matmul_q4_0(float* y, const float* x,
                     const ct_block_q4_0* W, int I, int O) {
    int blk_per_I = (I + 31) / 32;

    for (int j = 0; j < O; j++) {
        const ct_block_q4_0* row = W + (int64_t)j * blk_per_I;
        float32x4_t vacc = vdupq_n_f32(0.0f);

        for (int b = 0; b < blk_per_I; b++) {
            float d = fp16_to_f32(row[b].d);
            float32x4_t dv = vdupq_n_f32(d);
            int i0 = b * 32;

            int8_t qs[32];
            for (int k = 0; k < 16; k++) {
                qs[k * 2]     = (int8_t)((row[b].qs[k] & 0x0F) - 8);
                qs[k * 2 + 1] = (int8_t)((row[b].qs[k] >> 4) - 8);
            }

            for (int g = 0; g < 4; g++) {
                int16x8_t w16 = vmovl_s8(vld1_s8(qs + g * 8));
                float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
                float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));

                int ii = i0 + g * 8;
                if (ii + 4 > I) break;
                float32x4_t x_lo = vld1q_f32(x + ii);
                float32x4_t x_hi = vld1q_f32(x + ii + 4);
                vacc = vfmaq_f32(vacc, x_lo, vmulq_f32(dv, wf_lo));
                vacc = vfmaq_f32(vacc, x_hi, vmulq_f32(dv, wf_hi));
            }
        }
        y[j] = vaddvq_f32(vacc);

        /* Scalar remainder for partial last block */
        int rem_start = blk_per_I * 32;
        if (rem_start > I) rem_start = I - 32;
        if (rem_start < 0) rem_start = 0;
        for (int i = rem_start; i < I; i++) {
            int b = i / 32;
            int k = i % 32;
            int nib = (row[b].qs[k >> 1] >> ((k & 1) << 2)) & 0xF;
            y[j] += x[i] * fp16_to_f32(row[b].d) * ((float)nib - 8.0f);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * NEON: BQ1_0 Matmul — binary {−1,+1} × scale
 *
 * W stored [O][I]: each output row j has ceil(I/128) groups.
 * Each group: 128 binary values + FP16 scale.
 * y[j] = Σ_g d_g * ( Σ_{i in group} x[i] * (bit_i ? +1 : -1) )
 * ═══════════════════════════════════════════════════════════════ */

void ct_matmul_bq1_0(float* y, const float* x,
                      const ct_block_bq1_0* W, int I, int O) {
    int ng_per_row = (I + 127) / 128;

    for (int j = 0; j < O; j++) {
        const ct_block_bq1_0* row = W + (int64_t)j * ng_per_row;
        float sum = 0.0f;

        for (int g = 0; g < ng_per_row; g++) {
            float d = fp16_to_f32(row[g].d);
            uint64_t bits0 = row[g].bits[0];
            uint64_t bits1 = row[g].bits[1];
            int i0 = g * 128;

            /* Accumulate dot product: x[i] * (±d) */
            int n = (I - i0 < 128) ? I - i0 : 128;
            int half_n = (n < 64) ? n : 64;

            for (int k = 0; k < half_n; k++) {
                int bit = (bits0 >> (uint64_t)k) & 1;
                sum += x[i0 + k] * (bit ? d : -d);
            }
            for (int k = 0; k < n - 64; k++) {
                int bit = (bits1 >> (uint64_t)k) & 1;
                sum += x[i0 + 64 + k] * (bit ? d : -d);
            }
        }
        y[j] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * NEON: Batch Matmul — y[n][O] = x[n][I] @ W^T
 *
 * For n_tokens > 1, processes all tokens against the same weight matrix.
 * For n_tokens == 1, falls through to the single-vector path.
 * ═══════════════════════════════════════════════════════════════ */

void ct_matmul_batch_q8_0(float* y, const float* x, int n_tokens,
                           const ct_block_q8_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_q8_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

void ct_matmul_batch_q4_0(float* y, const float* x, int n_tokens,
                           const ct_block_q4_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_q4_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

void ct_matmul_batch_bq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_bq1_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_bq1_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

void ct_matmul_batch_tq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_tq1_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_tq1_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

/* ═══════════════════════════════════════════════════════════════
 * F32 Matmul (reference)
 * ═══════════════════════════════════════════════════════════════ */

void ct_matmul_f32(float* y, const float* x, const float* W, int I, int O) {
    for (int o = 0; o < O; o++) {
        float32x4_t acc = vdupq_n_f32(0);
        const float* wrow = W + (int64_t)o * I;
        int i = 0;
        for (; i + 4 <= I; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(x + i), vld1q_f32(wrow + i));
        float s = vaddvq_f32(acc);
        for (; i < I; i++)
            s += x[i] * wrow[i];
        y[o] = s;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Init: call once at startup to build FP16 LUT
 * ═══════════════════════════════════════════════════════════════ */

void ct_quant_init(void) {
    ct_fp16_init();
}

#elif !defined(__AVX2__)  /* scalar fallback (no SIMD at all) */

/* ── Scalar implementations for non-ARM platforms ── */

static bool ct_fp16_inited = false;
void ct_quant_init(void) {
    if (!ct_fp16_inited) {
        ct_fp16_init();
        ct_fp16_inited = true;
    }
}

void ct_dequant_q8_0(const ct_block_q8_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 32 ? count : 32;
    for (int i = 0; i < n; i++) out[i] = block->qs[i] * d;
    for (int i = n; i < count; i++) out[i] = 0;
}

void ct_dequant_q4_0(const ct_block_q4_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 32 ? count : 32;
    for (int i = 0; i < n; i++) {
        int nib = (block->qs[i >> 1] >> ((i & 1) << 2)) & 0xF;
        out[i] = ((float)nib - 8.0f) * d;
    }
    for (int i = n; i < count; i++) out[i] = 0;
}

void ct_dequant_bq1_0(const ct_block_bq1_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 128 ? count : 128;
    for (int i = 0; i < n; i++) {
        int byte_idx = i >> 6;
        int bit_idx = i & 0x3F;
        out[i] = ((block->bits[byte_idx] >> bit_idx) & 1) ? d : -d;
    }
    for (int i = n; i < count; i++) out[i] = 0;
}

void ct_dequant_tq1_0(const ct_block_tq1_0* block, float* out, int count) {
    float d = fp16_to_f32(block->d);
    int n = count < 256 ? count : 256;
    int idx = 0;
    for (int i = 0; i < 48 && idx < n; i++) {
        int vals[5];
        tq1_unpack_byte(block->qs[i], vals);
        for (int j = 0; j < 5 && idx < n; j++)
            out[idx++] = (float)vals[j] * d;
    }
    for (int i = 0; i < 4 && idx < n; i++) {
        for (int j = 0; j < 4 && idx < n; j++) {
            int v = (block->qh[i] >> (j * 2)) & 3;
            out[idx++] = (float)(v - 1) * d;
        }
    }
    for (; idx < count; idx++) out[idx] = 0;
}

void ct_matmul_q8_0(float* y, const float* x, const ct_block_q8_0* W, int I, int O) {
    int blk_per_I = (I + 31) / 32;
    for (int j = 0; j < O; j++) {
        const ct_block_q8_0* row = W + (int64_t)j * blk_per_I;
        float sum = 0.0f;
        for (int b = 0; b < blk_per_I; b++) {
            float d = fp16_to_f32(row[b].d);
            int i0 = b * 32;
            int n = (I - i0 < 32) ? I - i0 : 32;
            for (int k = 0; k < n; k++)
                sum += x[i0 + k] * d * row[b].qs[k];
        }
        y[j] = sum;
    }
}

void ct_matmul_q4_0(float* y, const float* x, const ct_block_q4_0* W, int I, int O) {
    int blk_per_I = (I + 31) / 32;
    for (int j = 0; j < O; j++) {
        const ct_block_q4_0* row = W + (int64_t)j * blk_per_I;
        float sum = 0.0f;
        for (int b = 0; b < blk_per_I; b++) {
            float d = fp16_to_f32(row[b].d);
            int i0 = b * 32;
            int n = (I - i0 < 32) ? I - i0 : 32;
            for (int k = 0; k < n; k++) {
                int nib = (row[b].qs[k >> 1] >> ((k & 1) << 2)) & 0xF;
                sum += x[i0 + k] * ((float)nib - 8.0f) * d;
            }
        }
        y[j] = sum;
    }
}

void ct_matmul_bq1_0(float* y, const float* x, const ct_block_bq1_0* W, int I, int O) {
    int ng_per_row = (I + 127) / 128;
    for (int j = 0; j < O; j++) {
        const ct_block_bq1_0* row = W + (int64_t)j * ng_per_row;
        float sum = 0.0f;
        for (int g = 0; g < ng_per_row; g++) {
            float d = fp16_to_f32(row[g].d);
            uint64_t b0 = row[g].bits[0];
            uint64_t b1 = row[g].bits[1];
            int i0 = g * 128;
            int n = (I - i0 < 128) ? I - i0 : 128;
            for (int k = 0; k < n; k++) {
                uint64_t bits = (k < 64) ? b0 : b1;
                int bit = (bits >> (k & 0x3F)) & 1;
                sum += x[i0 + k] * (bit ? d : -d);
            }
        }
        y[j] = sum;
    }
}

void ct_matmul_tq1_0(float* y, const float* x, const ct_block_tq1_0* W, int I, int O) {
    int nb_per_row = (I + 255) / 256;
    for (int j = 0; j < O; j++) {
        const ct_block_tq1_0* row = W + (int64_t)j * nb_per_row;
        float sum = 0.0f;
        for (int b = 0; b < nb_per_row; b++) {
            float d = fp16_to_f32(row[b].d);
            int i0 = b * 256;
            int n = (I - i0 < 256) ? I - i0 : 256;
            int idx = 0;
            for (int j2 = 0; j2 < 48 && idx < n; j2++) {
                int vals[5];
                tq1_unpack_byte(row[b].qs[j2], vals);
                for (int k = 0; k < 5 && idx < n; k++) {
                    sum += x[i0 + idx] * (float)vals[k] * d;
                    idx++;
                }
            }
            for (int j2 = 0; j2 < 4 && idx < n; j2++) {
                for (int k = 0; k < 4 && idx < n; k++) {
                    int v = (row[b].qh[j2] >> (k * 2)) & 3;
                    if (v != 1)
                        sum += x[i0 + idx] * (float)((v == 2) ? d : -d);
                    idx++;
                }
            }
        }
        y[j] = sum;
    }
}

void ct_matmul_f32(float* y, const float* x, const float* W, int I, int O) {
    for (int j = 0; j < O; j++) {
        const float* row = W + (int64_t)j * I;
        float sum = 0.0f;
        for (int i = 0; i < I; i++)
            sum += x[i] * row[i];
        y[j] = sum;
    }
}

void ct_matmul_batch_q8_0(float* y, const float* x, int n_tokens,
                           const ct_block_q8_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_q8_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

void ct_matmul_batch_q4_0(float* y, const float* x, int n_tokens,
                           const ct_block_q4_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_q4_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

void ct_matmul_batch_bq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_bq1_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_bq1_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

void ct_matmul_batch_tq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_tq1_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_tq1_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

#endif /* __ARM_NEON / !__AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * x86 AVX2: Quantized Matmul Kernels
 *
 * Each function replaces the scalar fallback when compiled with
 * -mavx2 -mfma (automatically defines __AVX2__).
 * ═══════════════════════════════════════════════════════════════ */
#ifdef __AVX2__

#include <immintrin.h>

/* ─── Utility: horizontal sum of __m256 ─── */
static inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}

/* ─── AVX2: Q8_0 Matmul ─── */
void ct_matmul_q8_0(float* y, const float* x,
                     const ct_block_q8_0* W, int I, int O) {
    int blk_per_I = (I + 31) / 32;

    for (int j = 0; j < O; j++) {
        const ct_block_q8_0* row = W + (int64_t)j * blk_per_I;
        __m256 vacc = _mm256_setzero_ps();

        for (int b = 0; b < blk_per_I; b++) {
            float d = fp16_to_f32(row[b].d);
            __m256 dv = _mm256_set1_ps(d);
            int i0 = b * 32;

            for (int g = 0; g < 4; g++) {
                int ii = i0 + g * 8;
                if (ii + 8 > I) break;

                /* Load 8 int8 weights, sign-extend to int32, convert to float */
                __m128i w8 = _mm_loadl_epi64((const __m128i*)(row[b].qs + g * 8));
                __m128i w32_lo = _mm_cvtepi8_epi32(w8);
                __m128i w32_hi = _mm_cvtepi8_epi32(_mm_srli_si128(w8, 4));
                __m128 wf_lo = _mm_cvtepi32_ps(w32_lo);
                __m128 wf_hi = _mm_cvtepi32_ps(w32_hi);
                __m256 wf = _mm256_set_m128(wf_hi, wf_lo);

                __m256 xv = _mm256_loadu_ps(x + ii);
                vacc = _mm256_fmadd_ps(xv, _mm256_mul_ps(dv, wf), vacc);
            }
        }

        /* Horizontal sum */
        y[j] = hsum_ps(vacc);

        /* Scalar remainder for partial last block */
        int rem_start = blk_per_I * 32;
        if (rem_start > I) rem_start = I - 32;
        if (rem_start < 0) rem_start = 0;
        for (int i = rem_start; i < I; i++) {
            int b = i / 32;
            int k = i % 32;
            y[j] += x[i] * fp16_to_f32(row[b].d) * row[b].qs[k];
        }
    }
}

/* ─── AVX2: Q4_0 Matmul ─── */
void ct_matmul_q4_0(float* y, const float* x,
                     const ct_block_q4_0* W, int I, int O) {
    int blk_per_I = (I + 31) / 32;

    for (int j = 0; j < O; j++) {
        const ct_block_q4_0* row = W + (int64_t)j * blk_per_I;
        __m256 vacc = _mm256_setzero_ps();

        for (int b = 0; b < blk_per_I; b++) {
            float d = fp16_to_f32(row[b].d);
            __m256 dv = _mm256_set1_ps(d);
            int i0 = b * 32;

            /* Unpack nibbles to int8 (subtract 8 for signedness) */
            int8_t qs[32];
            for (int k = 0; k < 16; k++) {
                qs[k * 2]     = (int8_t)((row[b].qs[k] & 0x0F) - 8);
                qs[k * 2 + 1] = (int8_t)((row[b].qs[k] >> 4) - 8);
            }

            for (int g = 0; g < 4; g++) {
                int ii = i0 + g * 8;
                if (ii + 8 > I) break;

                __m128i w8 = _mm_loadl_epi64((const __m128i*)(qs + g * 8));
                __m128i w32_lo = _mm_cvtepi8_epi32(w8);
                __m128i w32_hi = _mm_cvtepi8_epi32(_mm_srli_si128(w8, 4));
                __m128 wf_lo = _mm_cvtepi32_ps(w32_lo);
                __m128 wf_hi = _mm_cvtepi32_ps(w32_hi);
                __m256 wf = _mm256_set_m128(wf_hi, wf_lo);

                __m256 xv = _mm256_loadu_ps(x + ii);
                vacc = _mm256_fmadd_ps(xv, _mm256_mul_ps(dv, wf), vacc);
            }
        }

        y[j] = hsum_ps(vacc);

        /* Scalar remainder for partial last block */
        int rem_start = blk_per_I * 32;
        if (rem_start > I) rem_start = I - 32;
        if (rem_start < 0) rem_start = 0;
        for (int i = rem_start; i < I; i++) {
            int b = i / 32;
            int k = i % 32;
            int nib = (row[b].qs[k >> 1] >> ((k & 1) << 2)) & 0xF;
            y[j] += x[i] * fp16_to_f32(row[b].d) * ((float)nib - 8.0f);
        }
    }
}

/* ─── AVX2: BQ1_0 Matmul (scalar — bit-by-bit, same as fallback) ─── */
void ct_matmul_bq1_0(float* y, const float* x,
                      const ct_block_bq1_0* W, int I, int O) {
    int ng_per_row = (I + 127) / 128;
    for (int j = 0; j < O; j++) {
        const ct_block_bq1_0* row = W + (int64_t)j * ng_per_row;
        float sum = 0.0f;
        for (int g = 0; g < ng_per_row; g++) {
            float d = fp16_to_f32(row[g].d);
            uint64_t bits0 = row[g].bits[0];
            uint64_t bits1 = row[g].bits[1];
            int i0 = g * 128;
            int n = (I - i0 < 128) ? I - i0 : 128;
            int half_n = (n < 64) ? n : 64;
            for (int k = 0; k < half_n; k++)
                sum += x[i0 + k] * ((bits0 >> k) & 1 ? d : -d);
            for (int k = 0; k < n - 64; k++)
                sum += x[i0 + 64 + k] * ((bits1 >> k) & 1 ? d : -d);
        }
        y[j] = sum;
    }
}

/* ─── AVX2: TQ1_0 Matmul (scalar — ternary decode, same as fallback) ─── */
void ct_matmul_tq1_0(float* y, const float* x,
                      const ct_block_tq1_0* W, int I, int O) {
    int nb_per_row = (I + 255) / 256;
    for (int j = 0; j < O; j++) {
        const ct_block_tq1_0* row = W + (int64_t)j * nb_per_row;
        float sum = 0.0f;
        for (int b = 0; b < nb_per_row; b++) {
            float d = fp16_to_f32(row[b].d);
            int i0 = b * 256;
            int idx = 0;
            int n = (I - i0 < 256) ? I - i0 : 256;
            for (int j2 = 0; j2 < 48 && idx < n; j2++) {
                int tmp = row[b].qs[j2];
                for (int k = 0; k < 5 && idx < n; k++) {
                    int v = (tmp % 3) - 1;
                    tmp /= 3;
                    if (v != 0) sum += x[i0 + idx] * d * (float)v;
                    idx++;
                }
            }
            for (int j2 = 0; j2 < 4 && idx < n; j2++) {
                uint8_t byte_val = row[b].qh[j2];
                for (int k = 0; k < 4 && idx < n; k++) {
                    int v = (byte_val >> (k * 2)) & 3;
                    if (v != 1) {
                        float sv = (v == 2) ? 1.0f : -1.0f;
                        sum += x[i0 + idx] * d * sv;
                    }
                    idx++;
                }
            }
        }
        y[j] = sum;
    }
}

/* ─── AVX2: F32 Matmul (simple, no SIMD needed) ─── */
void ct_matmul_f32(float* y, const float* x, const float* W, int I, int O) {
    for (int o = 0; o < O; o++) {
        const float* wrow = W + (int64_t)o * I;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= I; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(wrow + i), acc);
        float sum = hsum_ps(acc);
        for (; i < I; i++) sum += x[i] * wrow[i];
        y[o] = sum;
    }
}

/* ─── AVX2: Batch Matmul Wrappers ─── */
void ct_matmul_batch_q8_0(float* y, const float* x, int n_tokens,
                           const ct_block_q8_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_q8_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}
void ct_matmul_batch_q4_0(float* y, const float* x, int n_tokens,
                           const ct_block_q4_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_q4_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}
void ct_matmul_batch_bq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_bq1_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_bq1_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}
void ct_matmul_batch_tq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_tq1_0* W, int I, int O) {
    for (int t = 0; t < n_tokens; t++)
        ct_matmul_tq1_0(y + (int64_t)t * O, x + (int64_t)t * I, W, I, O);
}

#endif /* __AVX2__ */

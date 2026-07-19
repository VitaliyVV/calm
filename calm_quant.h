/**
 * calm_quant.h — Calm Quantized Tensor Kernels
 *
 * BQ1_0: binary 1-bit {−1,+1} with FP16 group scales (128 weights/group)
 * TQ1_0: ternary 1.58-bit {−1,0,+1} base-3 packing (256 weights/block)
 * Q8_0:  8-bit block quantization (32 elements/block, FP16 scale)
 * Q4_0:  4-bit block quantization (32 elements/block, FP16 scale)
 *
 * Для сборки: clang -O3 -std=c11 -march=native -DCT_NEON
 */
#ifndef CALM_QUANT_H
#define CALM_QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * Quant Block Structures
 * ═══════════════════════════════════════════════════════════════ */

/* Computed block sizes (may differ from sizeof due to struct padding) */
#define CT_SIZEOF_Q8_0  34   /* uint16 + int8[32] */
#define CT_SIZEOF_Q4_0  18   /* uint16 + uint8[16] */
#define CT_SIZEOF_BQ1_0 18   /* uint64[2] + uint16 */
#define CT_SIZEOF_TQ1_0 54   /* uint8[48] + uint8[4] + uint16 */

/* Q8_0: 32 elements per block, FP16 scale + 8-bit values */
#define CT_QK8_0 32
typedef struct {
    uint16_t d;        /* FP16 scale (delta) */
    int8_t   qs[32];   /* quantized values */
} __attribute__((packed)) ct_block_q8_0;

/* Q4_0: 32 elements per block, FP16 scale + packed nibbles */
#define CT_QK4_0 32
typedef struct {
    uint16_t d;        /* FP16 scale (delta) */
    uint8_t  qs[16];   /* packed nibbles (2 values per byte) */
} __attribute__((packed)) ct_block_q4_0;

/* BQ1_0: binary 1-bit, 128 weights per group
 * 128 weights packed as 128 bits (2× uint64), 1 FP16 scale per group
 * Weight_i = (bit_i == 1) ? +scale : -scale
 */
#define CT_BQ1_0_GROUP_SIZE 128
typedef struct {
    uint64_t bits[2];  /* 128 bits: bit=0 → −scale, bit=1 → +scale */
    uint16_t d;         /* FP16 scale for this group */
} __attribute__((packed)) ct_block_bq1_0;

/* TQ1_0: ternary 1.58-bit, 256 weights per block
 * base-3 packing: 5 ternary values per byte (3^5=243 fits in 1 byte)
 * 256 values = 240 (48 bytes) + 16 (4 bytes) packing
 * Plus FP16 scale
 */
#define CT_TQ1_0_BLOCK_SIZE 256
typedef struct {
    uint8_t  qs[48];    /* (256-16)/5 = 48 bytes: 240 values in base-3 */
    uint8_t  qh[4];     /* 4 bytes: remaining 16 values (4 per byte) */
    uint16_t d;          /* FP16 scale */
} __attribute__((packed)) ct_block_tq1_0;

/* ═══════════════════════════════════════════════════════════════
 * FP16 conversion (inlined for performance)
 * ═══════════════════════════════════════════════════════════════ */

static inline float ct_fp16_to_fp32(uint16_t h) {
    /* Return 0 for NaN/Inf (exp=31) to prevent model corruption cascade */
    if (((h >> 10) & 0x1f) == 0x1f) return 0.0f;
    const uint32_t sign  = ((uint32_t)h & 0x8000U) << 16;
    const uint32_t exp16 = ((uint32_t)h >> 10) & 0x1FU;
    const uint32_t mant  = (uint32_t)h & 0x03FFU;
    uint32_t r;
    if (exp16 == 0) {
        if (mant == 0) {
            r = sign;                           /* ±0 */
        } else {
            /* Subnormal: normalize */
            int shift = 10;
            uint32_t m = mant;
            while ((m & 0x0400) == 0) { m <<= 1; shift--; }
            m &= 0x03FF;
            uint32_t exp32 = (uint32_t)(127 - 24 + shift);  /* 103 + shift */
            r = sign | (exp32 << 23) | (m << 13);
        }
    } else if (exp16 == 31) {
        r = sign | 0x7F800000U | (mant << 13);  /* inf/nan */
    } else {
        uint32_t exp32 = exp16 + 127 - 15;
        r = sign | (exp32 << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &r, sizeof(f));
    return f;
}

static inline uint16_t ct_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t s = x & 0x80000000U;
    const uint32_t e = (x >> 23) & 0xFFU;
    const uint32_t m = x & 0x007FFFFFU;
    uint32_t h;
    if (e == 0) {
        h = s;  /* zero/subnormal → zero */
    } else if (e == 0xFF) {
        h = (s | 0x7C00U) | (m >> 13);  /* inf/nan */
    } else {
        int ne = (int)e - 127 + 15;
        if (ne >= 31) {
            h = s | 0x7C00U;  /* overflow → inf */
        } else if (ne <= 0) {
            /* subnormal: flush to zero for now */
            h = s;
        } else {
            h = s | ((uint32_t)ne << 10) | (m >> 13);
            /* round to nearest even */
            if ((m >> 12) & 1) {
                h += 1;
            }
        }
    }
    return (uint16_t)h;
}

/* ═══════════════════════════════════════════════════════════════
 * Dequantization (for reference/fallback)
 * ═══════════════════════════════════════════════════════════════ */

/* Initialize lookup tables (FP16 conversion, etc.) */
void ct_quant_init(void);

void ct_dequant_q8_0(const ct_block_q8_0* block, float* out, int count);
void ct_dequant_q4_0(const ct_block_q4_0* block, float* out, int count);
void ct_dequant_bq1_0(const ct_block_bq1_0* block, float* out, int count);
void ct_dequant_tq1_0(const ct_block_tq1_0* block, float* out, int count);

/* ═══════════════════════════════════════════════════════════════
 * Quantization (float → quantized)
 * ═══════════════════════════════════════════════════════════════ */

void ct_quant_q8_0(const float* x, ct_block_q8_0* block, int count);
void ct_quant_q4_0(const float* x, ct_block_q4_0* block, int count);
void ct_quant_bq1_0(const float* x, ct_block_bq1_0* block, int count);
void ct_quant_tq1_0(const float* x, ct_block_tq1_0* block, int count);

/* Fast path: original mean(|w|) quantize without PTQ calibration */
void ct_quant_tq1_0_fast(const float* x, ct_block_tq1_0* block, int count);

/* ═══════════════════════════════════════════════════════════════
 * PTQ Calibration (weight-only MSE-optimal scale/threshold)
 * ═══════════════════════════════════════════════════════════════ */

/* Calibrate binary quantization scale: finds α minimizing MSE
 *   BQ1_0: quant(w) = α·sign(w)
 *   Optimal α = mean(|w|) ← already what ct_quant_bq1_0 uses.
 * Returns the MSE-optimal scale (for verification).
 */
float ct_calibrate_binary(const float* x, int n, float* out_mse);

/* Calibrate ternary quantization: finds α minimizing MSE
 *   TQ1_0: quant(w) = 0 if |w| < α/2, else α·sign(w)
 *   Solves by sorting |w| and sweeping split points.
 * Returns the MSE-optimal scale.
 * If out_mse is non-NULL, stores the corresponding MSE.
 */
float ct_calibrate_ternary(const float* x, int n, float* out_mse);

/* ═══════════════════════════════════════════════════════════════
 * Matmul: y[O] = x[I] @ W^T where W is quantized [O × I]
 *
 * y:    output vector (O floats)
 * x:    input vector  (I floats)
 * W:    quantized weight matrix
 * I, O: dimensions (I = input cols, O = output rows)
 * ═══════════════════════════════════════════════════════════════ */

/* F32 matmul (reference) */
void ct_matmul_f32(float* y, const float* x, const float* W, int I, int O);

/* Q8_0 matmul (NEON-optimized) */
void ct_matmul_q8_0(float* y, const float* x,
                     const ct_block_q8_0* W, int I, int O);

/* Q4_0 matmul (NEON-optimized) */
void ct_matmul_q4_0(float* y, const float* x,
                     const ct_block_q4_0* W, int I, int O);

/* BQ1_0 matmul: binary {−1,+1} with group scaling (NEON-optimized) */
void ct_matmul_bq1_0(float* y, const float* x,
                      const ct_block_bq1_0* W, int I, int O);

/* TQ1_0 matmul: ternary {−1,0,+1} base-3 packing (NEON-optimized) */
void ct_matmul_tq1_0(float* y, const float* x,
                      const ct_block_tq1_0* W, int I, int O);

/* ═══════════════════════════════════════════════════════════════
 * Optimized batch matmul: y[n][O] = x[n][I] @ W^T
 * Processes n_tokens input vectors against the same weight matrix.
 * ═══════════════════════════════════════════════════════════════ */

void ct_matmul_batch_q8_0(float* y, const float* x, int n_tokens,
                           const ct_block_q8_0* W, int I, int O);
void ct_matmul_batch_q4_0(float* y, const float* x, int n_tokens,
                           const ct_block_q4_0* W, int I, int O);
void ct_matmul_batch_bq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_bq1_0* W, int I, int O);
void ct_matmul_batch_tq1_0(float* y, const float* x, int n_tokens,
                            const ct_block_tq1_0* W, int I, int O);

/* ═══════════════════════════════════════════════════════════════
 * Data size helpers
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t ct_q8_0_size(int rows, int cols) {
    int nblocks = (cols + CT_QK8_0 - 1) / CT_QK8_0;
    return (size_t)rows * nblocks * sizeof(ct_block_q8_0);
}
static inline size_t ct_q4_0_size(int rows, int cols) {
    int nblocks = (cols + CT_QK4_0 - 1) / CT_QK4_0;
    return (size_t)rows * nblocks * sizeof(ct_block_q4_0);
}
static inline size_t ct_bq1_0_size(int rows, int cols) {
    int ngroups = (cols + CT_BQ1_0_GROUP_SIZE - 1) / CT_BQ1_0_GROUP_SIZE;
    return (size_t)rows * ngroups * sizeof(ct_block_bq1_0);
}
static inline size_t ct_tq1_0_size(int rows, int cols) {
    int nblocks = (cols + CT_TQ1_0_BLOCK_SIZE - 1) / CT_TQ1_0_BLOCK_SIZE;
    return (size_t)rows * nblocks * sizeof(ct_block_tq1_0);
}

#ifdef __cplusplus
}
#endif

#endif /* CALM_QUANT_H */

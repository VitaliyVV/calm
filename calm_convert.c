/**
 * calm_convert.c — Phase 6: GGUF→GGUF Streaming Requantizer
 *
 * Преобразует существующие GGUF модели в BQ1_0 (binary 1-bit)
 * или TQ1_0 (ternary 1.58-bit) формат через row-by-row mmap streaming.
 *
 * Ключевое отличие от v0.1: не загружает ВСЕ тензоры в RAM.
 * Вместо этого — 2 прохода:
 *   Pass 1: читаем mmap-заголовок, вычисляем выходные размеры
 *   Pass 2: stream-конвертация тензор за тензором, строка за строкой
 *
 * Пиковая RAM = O(cols * 4) для 1 строки float ≈ 2 MB для 9B модели.
 *
 * Сборка:
 *   clang -O3 -std=c11 -march=native -o calm_convert calm_convert.c calm_gguf.c calm_quant.c -lm
 *
 * Использование:
 *   calm_convert --input model.gguf --format tq1_0 --output model-tq1_0.gguf
 *   calm_convert --input model.gguf --format bq1_0 --calibrate --output model-bq1_0.gguf
 */
#define _GNU_SOURCE
#include "calm_gguf.h"
#include "calm_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <pthread.h>

/* ═══════════════════════════════════════════════════════════════
 * Tensor classification
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    CT_TENSOR_SKIP,      /* Keep as-is (norms, tiny tensors) */
    CT_TENSOR_WEIGHT,    /* Quantize to 1-bit/ternary */
    CT_TENSOR_HEAD,      /* Keep in Q8_0 (token_embd, output) */
} ct_tensor_class;

static bool should_skip(const char* name) {
    if (!name) return false;
    if (strstr(name, "norm") || strstr(name, "_norm") ||
        strstr(name, ".bias") || strstr(name, "rope") ||
        strstr(name, "freqs")) {
        return true;
    }
    return false;
}

static bool is_head_tensor(const char* name) {
    if (!name) return false;
    /* Preserve only the actual embedding/output head weights in Q8_0.
     * Use exact name matching — NOT strstr — to avoid false positives
     * like blk.N.attn_output.weight (which contains "output" as substring). */
    if (strcmp(name, "token_embd.weight")  == 0 ||
        strcmp(name, "tok_embd.weight")    == 0 ||
        strcmp(name, "output.weight")      == 0 ||
        strcmp(name, "head.weight")        == 0) {
        return true;
    }
    return false;
}

/* Extract layer number from tensor name like "blk.7.attn_q.weight" → 7.
 * Returns -1 if the name doesn't match blk.N.* pattern. */
static int get_layer_number(const char* name) {
    if (!name) return -1;
    if (strncmp(name, "blk.", 4) != 0) return -1;
    const char* p = name + 4;
    int n = 0;
    while (*p >= '0' && *p <= '9') {
        n = n * 10 + (*p - '0');
        p++;
    }
    if (*p != '.') return -1;
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * K-Quant Dequantization (proper dequant for all K-quant types)
 *
 * These are llama.cpp-compatible dequantizers that convert
 * K-quant blocks → float for re-quantization.
 * ═══════════════════════════════════════════════════════════════ */

/* Q2_K: 2-bit, 256-element super-blocks (block_q2_K, llama.cpp)
 *   16 sub-blocks of 16 elements, each with 4-bit scale + 4-bit min
 *   Structure: scales[16] + qs[64] + d(F16) + dmin(F16) = 84 bytes
 *   qs: stride-32 interleave — byte (g*32+r) holds 4 values, column c at bits (c*2).
 */
static void dequant_q2_k(const uint8_t* data, float* out, int cols, int blk_idx) {
    const uint8_t* scales = data;          /* 16 bytes: 4-bit scale + 4-bit min per sub-block */
    const uint8_t* qs = data + 16;         /* 64 bytes: 2-bit quants */
    float d    = ct_fp16_to_fp32(*(const uint16_t*)(data + 80)); /* d last (llama.cpp) */
    float dmin = ct_fp16_to_fp32(*(const uint16_t*)(data + 82)); /* dmin last */
    int base = blk_idx * 256;

    for (int sb = 0; sb < 16; sb++) {
        float dl = d    * (float)(scales[sb] & 0x0F);
        float ml = dmin * (float)(scales[sb] >> 4);
        for (int j = 0; j < 16; j++) {
            int idx = base + sb * 16 + j;
            if (idx >= cols) return;
            int g = (sb * 16 + j) / 128;        /* 0 or 1 */
            int r = (sb * 16 + j) % 32;         /* row within group */
            int c = ((sb * 16 + j) % 128) / 32; /* column 0..3 */
            int qv = (qs[g * 32 + r] >> (c * 2)) & 3;
            out[idx] = (float)qv * dl - ml;
        }
    }
}

/* Q3_K: 3-bit, 256-element super-blocks (block_q3_K, llama.cpp)
 *   Layout: hmask[32] + qs[64] + scales[12] + d(F16,2B) = 110B
 *   qs: 2-bit low bits, stride-32 interleave (4 values per byte, 2 bits each)
 *   hmask: 1-bit high bit per element (bit=0 → subtract 4, making 3-bit signed)
 *   scales[12]: 16 × 6-bit values unpacked to int8, centered at 32
 *   d: FP16 super-block scale (first 2 bytes)
 */
static void dequant_q3_k(const uint8_t* data, float* out, int cols, int blk_idx) {
    const uint8_t* hm = data;              /* hmask: 32 bytes (first) */
    const uint8_t* q = data + 32;          /* qs: 64 bytes */
    const uint8_t* scales = data + 96;     /* scales: 12 bytes */
    float d_all = ct_fp16_to_fp32(*(const uint16_t*)(data + 108)); /* d last */
    int base = blk_idx * 256;

    /* Unpack scales[12] → 16 int8 values (llama.cpp algorithm) */
    uint32_t aux[4];
    memcpy(aux, scales, 12);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* sc = (const int8_t*)aux;

    int is = 0;
    uint8_t m = 1; /* bit advances across BOTH 128-groups: 1,2,4,8,16,32,64,128 */
    for (int g = 0; g < 2; g++) {
        int shift = 0;
        int q_off = g * 32;
        for (int j = 0; j < 4; j++) {
            float dl = d_all * (float)(sc[is++] - 32);
            for (int l = 0; l < 16; l++) {
                int idx = base + g * 128 + j * 32 + l;
                if (idx >= cols) return;
                int low = (q[q_off + l] >> shift) & 3;
                int high = (hm[l] & m) ? 1 : 0;
                out[idx] = dl * (float)(low - (high ? 0 : 4));
            }
            dl = d_all * (float)(sc[is++] - 32);
            for (int l = 0; l < 16; l++) {
                int idx = base + g * 128 + j * 32 + 16 + l;
                if (idx >= cols) return;
                int low = (q[q_off + 16 + l] >> shift) & 3;
                int high = (hm[16 + l] & m) ? 1 : 0;
                out[idx] = dl * (float)(low - (high ? 0 : 4));
            }
            shift += 2;
            m <<= 1;
        }
    }
}

/* Q8_K: 8-bit, 256-element super-blocks (llama.cpp block_q8_K)
 *   Layout: d(F32,4B) + qs[256] + bsums[16] = 292 bytes
 *   Single scale d for all 256 values
 */
static void dequant_q8_k(const uint8_t* data, float* out, int cols, int blk_idx) {
    float d = *(const float*)(data);
    const int8_t* qs = (const int8_t*)(data + 4);
    int base = blk_idx * 256;
    for (int j = 0; j < 256; j++) {
        int idx = base + j;
        if (idx >= cols) return;
        out[idx] = (float)qs[j] * d;
    }
}

/* Q4_K: 4-bit, 256-element super-blocks (block_q4_K, llama.cpp)
 *   Layout: d(F16,2B) dmin(F16,2B) scales[12] qs[128] = 144 bytes
 *   4 groups of 64; each group: 32 low-nibble + 32 high-nibble values
 *   scales[12]: 8 × 6-bit (scale+min) pairs via k4_get_scale_min, NO -16 offset
 */
static inline void k4_get_scale_min(int j, const uint8_t* q,
                                    uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void dequant_q4_k(const uint8_t* data, float* out, int cols, int blk_idx) {
    float d    = ct_fp16_to_fp32(*(const uint16_t*)(data + 0));
    float dmin = ct_fp16_to_fp32(*(const uint16_t*)(data + 2));
    const uint8_t* scales = data + 4;   /* 12 bytes */
    const uint8_t* q = data + 16;       /* 128 bytes: 4-bit, 2 per byte */
    int base = blk_idx * 256;

    int is = 0;
    for (int g = 0; g < 4; g++) {
        uint8_t sc, m;
        k4_get_scale_min(is + 0, scales, &sc, &m);
        float d1 = d * (float)sc; float m1 = dmin * (float)m;
        k4_get_scale_min(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc; float m2 = dmin * (float)m;
        for (int l = 0; l < 32; l++) {
            int idx = base + g * 64 + l;
            if (idx >= cols) return;
            out[idx] = d1 * (float)(q[l] & 0xF) - m1;
        }
        for (int l = 0; l < 32; l++) {
            int idx = base + g * 64 + 32 + l;
            if (idx >= cols) return;
            out[idx] = d2 * (float)(q[l] >> 4) - m2;
        }
        q += 32; is += 2;
    }
}

/* Q5_K: 5-bit, 256-element super-blocks (block_q5_K, llama.cpp)
 *   Layout: d(F16,2B) dmin(F16,2B) scales[12] qh[16] qs[128] = 176 bytes
 *   4 groups of 64; qh: 1 high bit per element (u1/u2 masks shift by 2 per group)
 */
static void dequant_q5_k(const uint8_t* data, float* out, int cols, int blk_idx) {
    float d    = ct_fp16_to_fp32(*(const uint16_t*)(data + 0));
    float dmin = ct_fp16_to_fp32(*(const uint16_t*)(data + 2));
    const uint8_t* scales = data + 4;   /* 12 bytes */
    const uint8_t* qh = data + 16;      /* 16 bytes: 1 high bit per element */
    const uint8_t* ql = data + 48;      /* 128 bytes: 4-bit low nibbles */
    int base = blk_idx * 256;

    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int g = 0; g < 4; g++) {
        uint8_t sc, m;
        k4_get_scale_min(is + 0, scales, &sc, &m);
        float d1 = d * (float)sc; float m1 = dmin * (float)m;
        k4_get_scale_min(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc; float m2 = dmin * (float)m;
        for (int l = 0; l < 32; l++) {
            int idx = base + g * 64 + l;
            if (idx >= cols) return;
            out[idx] = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; l++) {
            int idx = base + g * 64 + 32 + l;
            if (idx >= cols) return;
            out[idx] = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        }
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

/* Q6_K: 6-bit, 256-element super-blocks (block_q6_K, llama.cpp)
 *   Layout: ql[128] qh[64] scales[16] d(F16,2B) = 210 bytes
 *   2 groups of 128; strided quads: q1=(ql[l]&0xF)|((qh[l]>>0)&3)<<4, ... all -32
 *   per 128-group: ql+=64, qh+=32, sc+=8
 */
static void dequant_q6_k(const uint8_t* data, float* out, int cols, int blk_idx) {
    const uint8_t* ql = data;                          /* 128 bytes: 4-bit nibbles */
    const uint8_t* qh = data + 128;                    /* 64 bytes: 2-bit upper */
    const int8_t*  sc = (const int8_t*)(data + 192);   /* 16 int8 scales */
    float d = ct_fp16_to_fp32(*(const uint16_t*)(data + 208));
    int base = blk_idx * 256;

    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = ((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = ((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            if (base + n + l >= cols) return;
            out[base + n + l +  0] = d * (float)sc[is + 0] * (float)q1;
            out[base + n + l + 32] = d * (float)sc[is + 2] * (float)q2;
            out[base + n + l + 64] = d * (float)sc[is + 4] * (float)q3;
            out[base + n + l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        ql += 64; qh += 32; sc += 8;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Row dequantizer: dispatches by type
 * ═══════════════════════════════════════════════════════════════ */

static void dequant_row(const void* data, int src_type, float* out, int cols) {
    switch (src_type) {
        case CT_GGUF_TYPE_F32:
            memcpy(out, data, (size_t)cols * 4);
            return;
        case CT_GGUF_TYPE_F16: {
            const uint16_t* f16 = (const uint16_t*)data;
            for (int i = 0; i < cols; i++)
                out[i] = ct_fp16_to_fp32(f16[i]);
            return;
        }
        case CT_GGUF_TYPE_Q8_0: {
            int blk = (cols + 31) / 32;
            for (int b = 0; b < blk; b++) {
                const ct_block_q8_0* block = &((const ct_block_q8_0*)data)[b];
                float d = ct_fp16_to_fp32(block->d);
                for (int j = 0; j < 32 && b * 32 + j < cols; j++)
                    out[b * 32 + j] = block->qs[j] * d;
            }
            return;
        }
        case CT_GGUF_TYPE_Q4_0: {
            int blk = (cols + 31) / 32;
            for (int b = 0; b < blk; b++) {
                const ct_block_q4_0* block = &((const ct_block_q4_0*)data)[b];
                float d = ct_fp16_to_fp32(block->d);
                for (int j = 0; j < 32 && b * 32 + j < cols; j++) {
                    int nib = (block->qs[j >> 1] >> ((j & 1) << 2)) & 0xF;
                    out[b * 32 + j] = ((float)nib - 8.0f) * d;
                }
            }
            return;
        }
        case CT_GGUF_TYPE_Q4_1: {
            /* Q4_1: similar to Q4_0 but with nonzero offset */
            const struct { uint16_t d; uint16_t m; uint8_t qs[16]; }* blk;
            blk = (const void*)data;
            int nblk = (cols + 31) / 32;
            for (int b = 0; b < nblk; b++) {
                float d = ct_fp16_to_fp32(blk[b].d);
                float m = ct_fp16_to_fp32(blk[b].m);
                for (int j = 0; j < 32 && b * 32 + j < cols; j++) {
                    int nib = (blk[b].qs[j >> 1] >> ((j & 1) << 2)) & 0xF;
                    out[b * 32 + j] = (float)nib * d + m;
                }
            }
            return;
        }
        case CT_GGUF_TYPE_Q5_0: {
            /* Q5_0: 32 elements/block, FP16 scale + uint32_t high bits + 4-bit nibbles
             * Layout: d(F16)=2B, qh(uint32_t)=4B, qs[16]=16B → 22 bytes/block
             * Dequant: val = ((nibble | (high_bit<<4)) - 16) * d
             */
            int nblk = (cols + 31) / 32;
            for (int b = 0; b < nblk; b++) {
                const uint8_t* bp = (const uint8_t*)data + (size_t)b * 22;
                float d = ct_fp16_to_fp32(*(const uint16_t*)bp);
                uint32_t qh;
                memcpy(&qh, bp + 2, 4);
                const uint8_t* qs = bp + 6;
                for (int j = 0; j < 32 && b * 32 + j < cols; j++) {
                    int nib = (qs[j >> 1] >> ((j & 1) << 2)) & 0xF;
                    int hi = (qh >> j) & 1;
                    out[b * 32 + j] = ((float)(nib | (hi << 4)) - 16.0f) * d;
                }
            }
            return;
        }
        case CT_GGUF_TYPE_Q5_1: {
            /* Q5_1: 32 elements/block, FP16 d + FP16 m + uint32_t qh + 4-bit nibbles */
            int nblk = (cols + 31) / 32;
            for (int b = 0; b < nblk; b++) {
                const uint8_t* bp = (const uint8_t*)data + (size_t)b * 24;
                float d = ct_fp16_to_fp32(*(const uint16_t*)bp);
                float m = ct_fp16_to_fp32(*(const uint16_t*)(bp + 2));
                uint32_t qh;
                memcpy(&qh, bp + 4, 4);
                const uint8_t* qs = bp + 8;
                for (int j = 0; j < 32 && b * 32 + j < cols; j++) {
                    int nib = (qs[j >> 1] >> ((j & 1) << 2)) & 0xF;
                    int hi = (qh >> j) & 1;
                    out[b * 32 + j] = (float)(nib | (hi << 4)) * d + m;
                }
            }
            return;
        }
        case CT_GGUF_TYPE_Q2_K: {
            /* Q2_K: 256-element super-blocks, 84 bytes each */
            int nblk = (cols + 255) / 256;
            for (int b = 0; b < nblk; b++)
                dequant_q2_k((const uint8_t*)data + (size_t)b * 84, out, cols, b);
            return;
        }
        case CT_GGUF_TYPE_Q3_K: {
            int nblk = (cols + 255) / 256;
            for (int b = 0; b < nblk; b++)
                dequant_q3_k((const uint8_t*)data + (size_t)b * 110, out, cols, b);
            return;
        }
        case CT_GGUF_TYPE_Q4_K: {
            int nblk = (cols + 255) / 256;
            for (int b = 0; b < nblk; b++)
                dequant_q4_k((const uint8_t*)data + (size_t)b * 144, out, cols, b);
            return;
        }
        case CT_GGUF_TYPE_Q5_K: {
            int nblk = (cols + 255) / 256;
            for (int b = 0; b < nblk; b++)
                dequant_q5_k((const uint8_t*)data + (size_t)b * 176, out, cols, b);
            return;
        }
        case CT_GGUF_TYPE_Q6_K: {
            int nblk = (cols + 255) / 256;
            for (int b = 0; b < nblk; b++)
                dequant_q6_k((const uint8_t*)data + (size_t)b * 210, out, cols, b);
            return;
        }
        case CT_GGUF_TYPE_Q8_K: {
            /* Q8_K: 256-element super-blocks, 292 bytes each */
            int nblk = (cols + 255) / 256;
            for (int b = 0; b < nblk; b++)
                dequant_q8_k((const uint8_t*)data + (size_t)b * 292, out, cols, b);
            return;
        }
        default: {
            /* Unknown type: zero out */
            for (int i = 0; i < cols; i++)
                out[i] = 0;
            fprintf(stderr, "  Warning: unsupported source type %d (cols=%d), zeroing\n",
                    src_type, cols);
            return;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Row quantizer: float → target type
 * ═══════════════════════════════════════════════════════════════ */

static void quantize_row(const float* x, int cols, void* dst, int dst_type,
                          bool calibrate_ternary) {
    switch (dst_type) {
        case CT_GGUF_TYPE_BQ1_0: {
            int ng = (cols + 127) / 128;
            ct_block_bq1_0* blk = (ct_block_bq1_0*)dst;
            for (int g = 0; g < ng; g++) {
                int count = (g + 1) * 128 <= cols ? 128 : cols - g * 128;
                ct_quant_bq1_0(x + g * 128, &blk[g], count);
            }
            return;
        }
        case CT_GGUF_TYPE_TQ1_0: {
            int nb = (cols + 255) / 256;
            ct_block_tq1_0* blk = (ct_block_tq1_0*)dst;
            for (int b = 0; b < nb; b++) {
                int count = (b + 1) * 256 <= cols ? 256 : cols - b * 256;
                if (calibrate_ternary)
                    ct_quant_tq1_0(x + b * 256, &blk[b], count);
                else
                    ct_quant_tq1_0_fast(x + b * 256, &blk[b], count);
            }
            return;
        }
        case CT_GGUF_TYPE_Q8_0: {
            int nblk = (cols + 31) / 32;
            ct_block_q8_0* blk = (ct_block_q8_0*)dst;
            for (int b = 0; b < nblk; b++) {
                int count = (b + 1) * 32 <= cols ? 32 : cols - b * 32;
                ct_quant_q8_0(x + b * 32, &blk[b], count);
            }
            return;
        }
        default:
            memcpy(dst, x, (size_t)cols * 4);
            return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Row size calculation (bytes per row)
 * ═══════════════════════════════════════════════════════════════ */

static size_t row_size_bytes(int cols, int type) {
    switch (type) {
        case CT_GGUF_TYPE_F32:   return (size_t)cols * 4;
        case CT_GGUF_TYPE_F16:   return (size_t)cols * 2;
        case CT_GGUF_TYPE_Q8_0: {
            int nb = (cols + 31) / 32;
            return (size_t)nb * CT_SIZEOF_Q8_0;
        }
        case CT_GGUF_TYPE_Q4_0: {
            int nb = (cols + 31) / 32;
            return (size_t)nb * CT_SIZEOF_Q4_0;
        }
        case CT_GGUF_TYPE_Q4_1: {
            int nb = (cols + 31) / 32;
            return (size_t)nb * 20;  // d(2)+m(2)+qs(16)
        }
        case CT_GGUF_TYPE_Q5_0: {
            int nb = (cols + 31) / 32;
            return (size_t)nb * 22;  // d(2)+qh(4)+qs(16)
        }
        case CT_GGUF_TYPE_Q5_1: {
            int nb = (cols + 31) / 32;
            return (size_t)nb * 24;  // d(2)+m(2)+qh(4)+qs(16)
        }
        case CT_GGUF_TYPE_Q4_K: {
            int nb = (cols + 255) / 256;
            return (size_t)nb * 144;
        }
        case CT_GGUF_TYPE_Q5_K: {
            int nb = (cols + 255) / 256;
            return (size_t)nb * 176;  // d(2)+dmin(2)+scales(12)+qh(16)+qs(128)
        }
        case CT_GGUF_TYPE_Q6_K: {
            int nb = (cols + 255) / 256;
            return (size_t)nb * 210;  // ql(128)+qh(64)+scales(16)+d(2)
        }
        case CT_GGUF_TYPE_Q3_K: {
            int nb = (cols + 255) / 256;
            return (size_t)nb * 110;  // hmask(32)+qs(64)+scales(12)+d(2)
        }
        case CT_GGUF_TYPE_Q2_K: {
            int nb = (cols + 255) / 256;
            return (size_t)nb * 84;   // scales(16)+qs(64)+d(2)+dmin(2)
        }
        case CT_GGUF_TYPE_Q8_K: {
            int nb = (cols + 255) / 256;
            return (size_t)nb * 292;  // d(4)+qs(256)+bsums(32)
        }
        case CT_GGUF_TYPE_BQ1_0: {
            int ng = (cols + 127) / 128;
            return (size_t)ng * sizeof(ct_block_bq1_0);
        }
        case CT_GGUF_TYPE_TQ1_0: {
            int nb = (cols + 255) / 256;
            return (size_t)nb * sizeof(ct_block_tq1_0);
        }
        default:
            return (size_t)cols * 4;  /* fallback: F32 */
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Compute output tensor size from dims + target type
 * ═══════════════════════════════════════════════════════════════ */

static size_t tensor_output_size(const ct_gguf_tensor_info* t, int dst_type) {
    int rows, cols;
    if (t->n_dims == 1) {
        rows = 1;
        cols = (int)t->dims[0];
    } else {
        rows = (int)t->dims[1];
        cols = (int)t->dims[0];
    }
    return (size_t)rows * row_size_bytes(cols, dst_type);
}

/* ═══════════════════════════════════════════════════════════════
 * Stream-convert one tensor: read rows from mmap → dequant → requant
 * ═══════════════════════════════════════════════════════════════ */

static bool stream_tensor(FILE* out,
                          const ct_gguf_context* src,
                          const ct_gguf_tensor_info* t,
                          int dst_type,
                          bool calibrate,
                          const char* format_name,
                          int tensor_idx, int n_tensors,
                          bool preserve_head) {
    int rows, cols;
    if (t->n_dims == 1) {
        rows = 1;
        cols = (int)t->dims[0];
    } else {
        rows = (int)t->dims[1];
        cols = (int)t->dims[0];
    }

    const char* tname = t->name;

    /* Classify tensor */
    ct_tensor_class tclass = CT_TENSOR_WEIGHT;
    size_t nelements = (size_t)rows * cols;
    if (should_skip(tname) && nelements < 100000)
        tclass = CT_TENSOR_SKIP;
    else if (is_head_tensor(tname) && preserve_head)
        tclass = CT_TENSOR_HEAD;

    /* Determine actual output type */
    int actual_dst;
    if (tclass == CT_TENSOR_SKIP)
        actual_dst = t->type;    /* keep original */
    else if (tclass == CT_TENSOR_HEAD)
        actual_dst = CT_GGUF_TYPE_Q8_0;
    else
        actual_dst = dst_type;

    size_t src_stride = row_size_bytes(cols, t->type);
    size_t dst_stride = row_size_bytes(cols, actual_dst);

    /* Progress */
    printf("  [%3d/%3d] %-40s %s (%d×%d, %s", tensor_idx + 1, n_tensors,
           tname,
           tclass == CT_TENSOR_SKIP ? "skip" :
           tclass == CT_TENSOR_HEAD ? "Q8_0" : format_name,
           rows, cols,
           ct_gguf_type_name(t->type));

    if (tclass == CT_TENSOR_SKIP) {
        /* Copy original row-by-row (for consistency, though we could memcpy) */
        for (int r = 0; r < rows; r++) {
            const void* src_row = ct_gguf_tensor_data(src, t) + (size_t)r * src_stride;
            fwrite(src_row, 1, src_stride, out);
        }
        printf(" → %s, %.1f MB)\n",
               ct_gguf_type_name(t->type),
               (rows * src_stride) / (1024.0 * 1024.0));
        return true;
    }

    /* Allocate row buffers */
    float* row_buf = (float*)malloc((size_t)cols * sizeof(float));
    void* dst_buf = malloc(dst_stride > 0 ? dst_stride : 1);
    if (!row_buf || !dst_buf) {
        free(row_buf); free(dst_buf);
        fprintf(stderr, "\n  Error: OOM for row buffers (%d cols)\n", cols);
        return false;
    }

    const void* tensor_base = ct_gguf_tensor_data(src, t);
    double peak_mb = 0;
    for (int r = 0; r < rows; r++) {
        /* Source row from mmap (zero-copy) */
        const void* src_row = (const uint8_t*)tensor_base + (size_t)r * src_stride;

        /* Validate row is within mmap bounds */
        if ((const uint8_t*)src_row + src_stride > (const uint8_t*)src->data + src->size) {
            fprintf(stderr, "\n  Error: tensor '%s' row %d exceeds mmap (offset=%zu, size=%zu, mmap=%zu)\n",
                    tname, r, (size_t)r * src_stride, src_stride, src->size);
            free(row_buf); free(dst_buf);
            return false;
        }

        /* Dequant → float */
        dequant_row(src_row, t->type, row_buf, cols);

        /* Quant → target format */
        quantize_row(row_buf, cols, dst_buf, actual_dst, calibrate);

        /* Write to output */
        fwrite(dst_buf, 1, dst_stride, out);

        /* Track peak */
        double mb = ((size_t)cols * sizeof(float) + dst_stride) / (1024.0 * 1024.0);
        if (mb > peak_mb) peak_mb = mb;
    }

    free(row_buf);
    free(dst_buf);

    printf(" → %s, %.1f MB, peak=%.1f MB)\n",
           ct_gguf_type_name(actual_dst),
           (rows * dst_stride) / (1024.0 * 1024.0),
           peak_mb);
    return true;
}

/* ═══════════════════════════════════════════════════════════════
 * GGUF Writer — Phase 1: write header + metadata + tensor info
 * ═══════════════════════════════════════════════════════════════ */

static bool write_header_and_metadata(FILE* f, const ct_gguf_context* src,
                                        const int* out_types,
                                        const size_t* out_sizes,
                                        const size_t* out_offsets,
                                        int n_tensors) {
    /* 1. Header */
    uint32_t magic = CT_GGUF_MAGIC;
    uint32_t version = 3;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);

    uint64_t n_tensors_u64 = (uint64_t)n_tensors;
    uint64_t n_metadata = src->metadata_count;
    fwrite(&n_tensors_u64, 8, 1, f);
    fwrite(&n_metadata, 8, 1, f);

    /* 2. Metadata (passthrough from source — re-encode) */
    for (uint64_t i = 0; i < src->metadata_count; i++) {
        size_t klen = strlen(src->metadata.keys[i]);
        uint64_t klen64 = (uint64_t)klen;
        fwrite(&klen64, 8, 1, f);
        fwrite(src->metadata.keys[i], 1, klen, f);

        uint32_t vtype = src->metadata.types[i];
        fwrite(&vtype, 4, 1, f);

        switch (vtype) {
            case CT_GGUF_VALUE_UINT8: {
                uint8_t v = src->metadata.values[i].v_u8;
                fwrite(&v, 1, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT8: {
                int8_t v = src->metadata.values[i].v_i8;
                fwrite(&v, 1, 1, f);
                break;
            }
            case CT_GGUF_VALUE_UINT16: {
                uint16_t v = src->metadata.values[i].v_u16;
                fwrite(&v, 2, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT16: {
                int16_t v = src->metadata.values[i].v_i16;
                fwrite(&v, 2, 1, f);
                break;
            }
            case CT_GGUF_VALUE_UINT32: {
                uint32_t v = src->metadata.values[i].v_u32;
                fwrite(&v, 4, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT32: {
                int32_t v = src->metadata.values[i].v_i32;
                fwrite(&v, 4, 1, f);
                break;
            }
            case CT_GGUF_VALUE_FLOAT32: {
                float v = src->metadata.values[i].v_f32;
                fwrite(&v, 4, 1, f);
                break;
            }
            case CT_GGUF_VALUE_BOOL: {
                uint8_t v = src->metadata.values[i].v_bool ? 1 : 0;
                fwrite(&v, 1, 1, f);
                break;
            }
            case CT_GGUF_VALUE_STRING: {
                const char* s = src->metadata.values[i].v_str;
                size_t slen = s ? strlen(s) : 0;
                uint64_t slen64 = (uint64_t)slen;
                fwrite(&slen64, 8, 1, f);
                if (slen > 0) fwrite(s, 1, slen, f);
                break;
            }
            case CT_GGUF_VALUE_ARRAY: {
                const char* k = src->metadata.keys[i];
                bool wrote = false;

                if (k && strcmp(k, "tokenizer.ggml.tokens") == 0 && src->vocab.n_vocab > 0) {
                    uint32_t arr_type = 8; /* string */
                    uint64_t arr_len = (uint64_t)src->vocab.n_vocab;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                    for (int vi = 0; vi < src->vocab.n_vocab; vi++) {
                        size_t sl = src->vocab.tokens[vi] ? strlen(src->vocab.tokens[vi]) : 0;
                        uint64_t sl64 = (uint64_t)sl;
                        fwrite(&sl64, 8, 1, f);
                        if (sl > 0) fwrite(src->vocab.tokens[vi], 1, sl, f);
                    }
                    wrote = true;
                }
                if (k && strcmp(k, "tokenizer.ggml.scores") == 0 && src->vocab.n_vocab > 0) {
                    uint32_t arr_type = 6; /* float32 */
                    uint64_t arr_len = (uint64_t)src->vocab.n_vocab;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                    for (int vi = 0; vi < src->vocab.n_vocab; vi++)
                        fwrite(&src->vocab.scores[vi], 4, 1, f);
                    wrote = true;
                }
                if (k && strcmp(k, "tokenizer.ggml.token_type") == 0 && src->vocab.n_vocab > 0) {
                    uint32_t arr_type = 5; /* int32 */
                    uint64_t arr_len = (uint64_t)src->vocab.n_vocab;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                    for (int vi = 0; vi < src->vocab.n_vocab; vi++)
                        fwrite(&src->vocab.token_types[vi], 4, 1, f);
                    wrote = true;
                }

                if (!wrote) {
                    uint32_t arr_type = 0;
                    uint64_t arr_len = 0;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                }
                break;
            }
            case CT_GGUF_VALUE_UINT64: {
                uint64_t v = src->metadata.values[i].v_u64;
                fwrite(&v, 8, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT64: {
                int64_t v = src->metadata.values[i].v_i64;
                fwrite(&v, 8, 1, f);
                break;
            }
            case CT_GGUF_VALUE_FLOAT64: {
                double v = src->metadata.values[i].v_f64;
                fwrite(&v, 8, 1, f);
                break;
            }
            default: break;
        }
    }

    /* 3. Tensor info — use pre-computed offsets from Pass 1 */
    for (int i = 0; i < n_tensors; i++) {
        const ct_gguf_tensor_info* t = &src->tensors[i];
        size_t tname_len = strlen(t->name);
        uint64_t tname_len64 = (uint64_t)tname_len;
        fwrite(&tname_len64, 8, 1, f);
        fwrite(t->name, 1, tname_len, f);

        uint32_t n_dims = t->n_dims;
        fwrite(&n_dims, 4, 1, f);

        for (uint32_t d = 0; d < n_dims; d++)
            fwrite(&t->dims[d], 8, 1, f);

        int actual_type = out_types ? out_types[i] : (int)t->type;
        fwrite(&actual_type, 4, 1, f);

        /* Use pre-computed 32-byte aligned offset from Pass 1 */
        size_t off = out_offsets ? out_offsets[i] : 0;
        fwrite(&off, 8, 1, f);
    }

    /* Pad to 32-byte alignment */
    size_t pos = ftell(f);
    while (pos % 32 != 0) {
        uint8_t pad = 0;
        fwrite(&pad, 1, 1, f);
        pos++;
    }

    return true;
}

/* ═══════════════════════════════════════════════════════════════
 * Post-conversion verification: open output GGUF and validate
 * ═══════════════════════════════════════════════════════════════ */

static bool verify_gguf(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "  Verify: cannot open %s\n", path); return false; }

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    size_t file_size = (size_t)st.st_size;

    /* mmap the output file for verification */
    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { fprintf(stderr, "  Verify: mmap failed\n"); return false; }

    /* Parse GGUF header */
    const uint8_t* buf = (const uint8_t*)data;
    if (buf[0] != 'G' || buf[1] != 'G' || buf[2] != 'U' || buf[3] != 'F') {
        fprintf(stderr, "  Verify: bad magic\n");
        munmap(data, file_size); return false;
    }

    uint64_t n_tensors;
    memcpy(&n_tensors, buf + 8, 8);
    uint64_t n_metadata;
    memcpy(&n_metadata, buf + 16, 8);

    /* Skip header + metadata to find tensor info start */
    size_t pos = 24;
    for (uint64_t m = 0; m < n_metadata; m++) {
        if (pos + 8 > file_size) goto verify_trunc;
        uint64_t klen; memcpy(&klen, buf + pos, 8); pos += 8;
        pos += (size_t)klen; /* skip key */
        if (pos + 4 > file_size) goto verify_trunc;
        uint32_t vtype; memcpy(&vtype, buf + pos, 4); pos += 4;
        switch (vtype) {
            case 0: case 1: case 7: pos += 1; break;
            case 2: case 3: pos += 2; break;
            case 4: case 5: case 6: pos += 4; break;
            case 8: {
                if (pos + 8 > file_size) goto verify_trunc;
                uint64_t sl; memcpy(&sl, buf + pos, 8); pos += 8;
                pos += (size_t)sl; break;
            }
            case 9: {
                uint32_t at; memcpy(&at, buf + pos, 4); pos += 4;
                uint64_t al; memcpy(&al, buf + pos + 4, 4); pos += 12;
                size_t elem_size = (at == 8) ? 0 : (at <= 3 ? 2 : (at <= 7 ? 4 : 8));
                if (at == 8) { /* array of strings — skip each */
                    for (uint64_t j = 0; j < al; j++) {
                        uint64_t sl; memcpy(&sl, buf + pos, 8); pos += 8;
                        pos += (size_t)sl;
                    }
                } else {
                    pos += (size_t)(al * elem_size);
                }
                break;
            }
            case 10: case 11: case 12: pos += 8; break;
            default: pos += 4; break;
        }
        if (pos > file_size) goto verify_trunc;
    }

    /* Parse tensor info and validate each tensor's data range */
    size_t errors = 0;
    size_t total_data = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (pos + 8 > file_size) { errors++; break; }
        uint64_t nl; memcpy(&nl, buf + pos, 8); pos += 8;
        char tname[256];
        size_t name_len = nl < 255 ? (size_t)nl : 255;
        memcpy(tname, buf + pos, name_len); tname[name_len] = '\0';
        pos += (size_t)nl;

        uint32_t nd; memcpy(&nd, buf + pos, 4); pos += 4;
        uint64_t dims[4] = {0};
        for (uint32_t d = 0; d < nd && d < 4; d++) {
            memcpy(&dims[d], buf + pos, 8); pos += 8;
        }

        uint32_t tt; memcpy(&tt, buf + pos, 4); pos += 4;
        uint64_t toff; memcpy(&toff, buf + pos, 8); pos += 8;

        /* Compute expected size */
        int rows = (nd > 1) ? (int)dims[1] : 1;
        int cols = (int)dims[0];
        size_t expected = row_size_bytes(cols, (int)tt) * (size_t)rows;
        total_data += expected;

        /* Align tensor data start (same as in write_header_and_metadata) */
        size_t data_start = 0;
        /* tensor_data_offset is computed below */
    }

    /* Tensor data section starts after all tensor info, 32-byte aligned */
    size_t tensor_data_off = (pos + 31) & ~(size_t)31;

    /* Second pass: validate with known data start */
    pos = 24; /* rewind for clean count */
    for (uint64_t m = 0; m < n_metadata; m++) {
        uint64_t kl; memcpy(&kl, buf + pos, 8); pos += 8;
        pos += (size_t)kl;
        uint32_t vt; memcpy(&vt, buf + pos, 4); pos += 4;
        switch (vt) {
            case 0: case 1: case 7: pos += 1; break;
            case 2: case 3: pos += 2; break;
            case 4: case 5: case 6: pos += 4; break;
            case 8: { uint64_t sl; memcpy(&sl, buf + pos, 8); pos += 8; pos += (size_t)sl; break; }
            case 9: {
                uint32_t at; memcpy(&at, buf + pos, 4); pos += 4;
                uint64_t al; memcpy(&al, buf + pos, 4); pos += 12;
                if (at == 8) { for (uint64_t j = 0; j < al; j++) { uint64_t sl; memcpy(&sl, buf + pos, 8); pos += 8; pos += (size_t)sl; } }
                else { size_t es = (at <= 3 ? 2 : (at <= 7 ? 4 : 8)); pos += (size_t)(al * es); }
                break;
            }
            case 10: case 11: case 12: pos += 8; break;
            default: pos += 4; break;
        }
    }

    errors = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nl; memcpy(&nl, buf + pos, 8); pos += 8;
        char tname[256];
        size_t name_len = nl < 255 ? (size_t)nl : 255;
        memcpy(tname, buf + pos, name_len); tname[name_len] = '\0';
        pos += (size_t)nl;

        uint32_t nd; memcpy(&nd, buf + pos, 4); pos += 4;
        uint64_t dims[4] = {0};
        for (uint32_t d = 0; d < nd && d < 4; d++) {
            memcpy(&dims[d], buf + pos, 8); pos += 8;
        }
        uint32_t tt; memcpy(&tt, buf + pos, 4); pos += 4;
        uint64_t toff; memcpy(&toff, buf + pos, 8); pos += 8;

        int rows = (nd > 1) ? (int)dims[1] : 1;
        int cols = (int)dims[0];
        size_t stride = row_size_bytes(cols, (int)tt);
        size_t tsize = stride * (size_t)rows;
        size_t abs_end = tensor_data_off + (size_t)toff + tsize;

        if (abs_end > file_size) {
            fprintf(stderr, "  Verify: '%s' %s overflows: offset=%llu end=%zu file=%zu (off by %zu)\n",
                    tname, ct_gguf_type_name((int)tt),
                    (unsigned long long)toff, abs_end, file_size, abs_end - file_size);
            errors++;
        }
    }

    munmap(data, file_size);

    if (errors == 0) {
        printf("  ✅ Verify: %llu tensors, all offsets valid (%zu bytes)\n",
               (unsigned long long)n_tensors, file_size);
        return true;
    }
    fprintf(stderr, "  ❌ Verify: %zu/%llu tensors FAILED offset check\n",
            errors, (unsigned long long)n_tensors);
    return false;

verify_trunc:
    fprintf(stderr, "  Verify: file truncated during parsing\n");
    munmap(data, file_size);
    return false;
}

/* ═══════════════════════════════════════════════════════════════
 * Parallel worker — multi-threaded tensor conversion
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int thread_id;
    int tensor_start;
    int tensor_end;
    const ct_gguf_context* src;
    int out_fd;                /* output fd for pwrite (thread-safe) */
    size_t data_start;         /* file offset where tensor data section begins */
    const size_t* out_offsets; /* pre-computed relative offsets */
    const int* out_types;
    int dst_type;
    bool calibrate;
    const char* format_name;
    bool preserve_head;
    volatile bool* global_ok;  /* shared atomic flag, set false on failure */
} worker_arg;

static void* worker_convert(void* arg) {
    worker_arg* w = (worker_arg*)arg;
    const ct_gguf_context* src = w->src;
    int out_fd = w->out_fd;
    size_t data_start = w->data_start;
    int n = w->tensor_end;

    for (int i = w->tensor_start; i < n; i++) {
        if (!*w->global_ok) return NULL;  /* abort on sibling failure */

        const ct_gguf_tensor_info* t = &src->tensors[i];
        int rows, cols;
        if (t->n_dims == 1) {
            rows = 1;
            cols = (int)t->dims[0];
        } else {
            rows = (int)t->dims[1];
            cols = (int)t->dims[0];
        }

        const char* tname = t->name;
        ct_tensor_class tclass = CT_TENSOR_WEIGHT;
        size_t nelements = (size_t)rows * cols;
        if (should_skip(tname) && nelements < 100000)
            tclass = CT_TENSOR_SKIP;
        else if (is_head_tensor(tname) && w->preserve_head)
            tclass = CT_TENSOR_HEAD;

        int actual_dst;
        if (tclass == CT_TENSOR_SKIP)     actual_dst = t->type;
        else if (tclass == CT_TENSOR_HEAD) actual_dst = CT_GGUF_TYPE_Q8_0;
        else                                actual_dst = w->dst_type;

        size_t src_stride = row_size_bytes(cols, t->type);
        size_t dst_stride = row_size_bytes(cols, actual_dst);

        /* Progress (lock-free — interleaved output is OK for terminal) */
        printf("  [%3d] %-40s %s (%d×%d, %s",
               i + 1, tname,
               tclass == CT_TENSOR_SKIP ? "skip" :
               tclass == CT_TENSOR_HEAD ? "Q8_0" : w->format_name,
               rows, cols,
               ct_gguf_type_name(t->type));

        size_t write_pos = data_start + w->out_offsets[i];

        if (tclass == CT_TENSOR_SKIP) {
            /* Copy original row-by-row */
            for (int r = 0; r < rows; r++) {
                const void* src_row = ct_gguf_tensor_data(src, t) + (size_t)r * src_stride;
                ssize_t written = pwrite(out_fd, src_row, src_stride, (off_t)(write_pos + (size_t)r * src_stride));
                if ((size_t)written != src_stride) {
                    fprintf(stderr, "\n  Error: pwrite failed for '%s' row %d\n", tname, r);
                    *w->global_ok = false; return NULL;
                }
            }
            printf(" → %s, %.1f MB)\n",
                   ct_gguf_type_name(t->type),
                   (rows * src_stride) / (1024.0 * 1024.0));
            continue;
        }

        /* Allocate row buffers */
        float* row_buf = (float*)malloc((size_t)cols * sizeof(float));
        void* dst_buf = malloc(dst_stride > 0 ? dst_stride : 1);
        if (!row_buf || !dst_buf) {
            free(row_buf); free(dst_buf);
            fprintf(stderr, "\n  Error: OOM for row buffers (%d cols)\n", cols);
            *w->global_ok = false; return NULL;
        }

        const void* tensor_base = ct_gguf_tensor_data(src, t);
        double peak_mb = 0;
        for (int r = 0; r < rows; r++) {
            const void* src_row = (const uint8_t*)tensor_base + (size_t)r * src_stride;
            if ((const uint8_t*)src_row + src_stride > (const uint8_t*)src->data + src->size) {
                fprintf(stderr, "\n  Error: tensor '%s' row %d exceeds mmap\n", tname, r);
                free(row_buf); free(dst_buf);
                *w->global_ok = false; return NULL;
            }
            dequant_row(src_row, t->type, row_buf, cols);
            quantize_row(row_buf, cols, dst_buf, actual_dst, w->calibrate);
            ssize_t written = pwrite(out_fd, dst_buf, dst_stride,
                                     (off_t)(write_pos + (size_t)r * dst_stride));
            if ((size_t)written != dst_stride) {
                fprintf(stderr, "\n  Error: pwrite failed for '%s' row %d\n", tname, r);
                free(row_buf); free(dst_buf);
                *w->global_ok = false; return NULL;
            }
            double mb = ((size_t)cols * sizeof(float) + dst_stride) / (1024.0 * 1024.0);
            if (mb > peak_mb) peak_mb = mb;
        }
        free(row_buf);
        free(dst_buf);

        printf(" → %s, %.1f MB, peak=%.1f MB)\n",
               ct_gguf_type_name(actual_dst),
               (rows * dst_stride) / (1024.0 * 1024.0),
               peak_mb);
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════
 * CLI + Main
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    const char* input_path = NULL;
    const char* output_path = NULL;
    const char* format = "tq1_0";
    bool calibrate = false;
    bool preserve_sensitive = true;
    bool verify = false;
    int adaptive_quant_layers = 0;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
            format = argv[++i];
        else if (strcmp(argv[i], "--calibrate") == 0)
            calibrate = true;
        else if (strcmp(argv[i], "--no-preserve") == 0)
            preserve_sensitive = false;
        else if (strcmp(argv[i], "--verify") == 0)
            verify = true;
        else if (strcmp(argv[i], "--adaptive-quant") == 0 && i + 1 < argc) {
            int n = atoi(argv[++i]);
            if (n < 0) n = 0;
            if (n > 999) n = 999;
            adaptive_quant_layers = n;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Calm Model Converter v0.2 — Phase 6 (Streaming Requantizer)\n");
            printf("  --input <file.gguf>      Source GGUF model\n");
            printf("  --format <bq1_0|tq1_0>   Target format (default: tq1_0)\n");
            printf("  --output <file.gguf>     Output path\n");
            printf("  --calibrate              MSE-optimal ternary calibration (slower, better quality)\n");
            printf("  --no-preserve            Don't preserve embed/output in Q8_0\n");
            printf("  --adaptive-quant <N>     Preserve first N layers in Q8_0 (rest in target format)\n");
            printf("  --verify                 Validate output GGUF after conversion\n");
            printf("  --help                   This help\n");
            printf("\n");
            printf("Peak RAM: ~row_buf + quant_buf ≈ few MB (streaming)\n");
            return 0;
        }
    }

    if (!input_path) { fprintf(stderr, "Error: --input required\n"); return 1; }
    if (!output_path) { fprintf(stderr, "Error: --output required\n"); return 1; }

    int dst_type;
    const char* format_name;
    if (strcmp(format, "bq1_0") == 0) {
        dst_type = CT_GGUF_TYPE_BQ1_0;
        format_name = "BQ1_0 (binary 1-bit)";
    } else if (strcmp(format, "tq1_0") == 0) {
        dst_type = CT_GGUF_TYPE_TQ1_0;
        format_name = "TQ1_0 (ternary 1.58-bit)";
    } else {
        fprintf(stderr, "Error: unknown format '%s'. Use bq1_0 or tq1_0.\n", format);
        return 1;
    }

    ct_quant_init();

    printf("Calm Model Converter v0.2 (Streaming Requantizer)\n");
    printf("  Input:  %s\n", input_path);
    printf("  Output: %s\n", output_path);
    printf("  Format: %s\n", format_name);
    if (calibrate) printf("  PTQ calibration: ON (MSE-optimal)\n");
    if (preserve_sensitive) printf("  Preserve embed/output: Q8_0\n");
    if (adaptive_quant_layers > 0) printf("  Adaptive quant: first %d layers in Q8_0, rest in %s\n",
                                           adaptive_quant_layers, format_name);
    printf("\n");

    /* Open source model (mmap) */
    ct_gguf_context* src = ct_gguf_open(input_path);
    if (!src) { fprintf(stderr, "Failed to open: %s\n", input_path); return 1; }

    printf("  Architecture: %s\n", ct_gguf_architecture(src));
    printf("  Tensors:      %llu\n", (unsigned long long)src->tensor_count);
    printf("  GGUF v%u\n", src->version);

    /* Print source format distribution */
    int type_counts[64] = {0};
    for (uint64_t i = 0; i < src->tensor_count; i++)
        if (src->tensors[i].type < 64)
            type_counts[src->tensors[i].type]++;
    printf("  Source types: ");
    for (int t = 0; t < 64; t++)
        if (type_counts[t] > 0)
            printf("%s=%d ", ct_gguf_type_name(t), type_counts[t]);
    printf("\n");
    printf("  Peak RAM mode: streaming (row-by-row, no full-tensor buffer)\n");
    printf("\n");

    int n = (int)src->tensor_count;

    /* === PASS 1: Compute output types and sizes === */
    int* out_types = malloc((size_t)n * sizeof(int));
    size_t* out_sizes = calloc((size_t)n, sizeof(size_t));
    if (!out_types || !out_sizes) {
        fprintf(stderr, "Error: OOM for tensor info\n");
        free(out_types); free(out_sizes);
        ct_gguf_close(src);
        return 1;
    }

    size_t total_input = 0;
    size_t total_output = 0;
    /* Pre-computed 32-byte aligned output offsets for each tensor */
    size_t* out_offsets = malloc((size_t)n * sizeof(size_t));
    if (!out_offsets) {
        fprintf(stderr, "Error: OOM for offsets\n");
        free(out_types); free(out_sizes);
        ct_gguf_close(src);
        return 1;
    }

    size_t running = 0;
    for (int i = 0; i < n; i++) {
        const ct_gguf_tensor_info* t = &src->tensors[i];
        const char* tname = t->name;
        int rows = (int)(t->n_dims > 1 ? t->dims[1] : 1);
        int cols = (int)t->dims[0];
        size_t nelements = (size_t)rows * cols;

        ct_tensor_class tclass = CT_TENSOR_WEIGHT;
        if (should_skip(tname) && nelements < 100000)
            tclass = CT_TENSOR_SKIP;
        else if (is_head_tensor(tname) && preserve_sensitive)
            tclass = CT_TENSOR_HEAD;
        else if (adaptive_quant_layers > 0) {
            int layer = get_layer_number(tname);
            if (layer >= 0 && layer < adaptive_quant_layers)
                tclass = CT_TENSOR_HEAD;  /* Preserve early layers in Q8_0 */
        }

        if (tclass == CT_TENSOR_SKIP)
            out_types[i] = t->type;
        else if (tclass == CT_TENSOR_HEAD)
            out_types[i] = CT_GGUF_TYPE_Q8_0;
        else
            out_types[i] = dst_type;

        out_sizes[i] = tensor_output_size(t, out_types[i]);
        out_offsets[i] = running;  /* 32-byte aligned */
        running += out_sizes[i];
        /* Align to 32 bytes for GGUF v3 spec */
        while (running % 32 != 0) running++;
        total_input += t->size;
        total_output += out_sizes[i];
    }

    printf("  Input size:  %.2f GB (%llu bytes)\n",
           total_input / (1024.0*1024.0*1024.0),
           (unsigned long long)total_input);
    printf("  Output size: %.2f GB (%llu bytes)\n",
           total_output / (1024.0*1024.0*1024.0),
           (unsigned long long)total_output);
    printf("  Compression: %.1f%%\n",
           total_input > 0 ? (100.0 * (1.0 - (double)total_output / (double)total_input)) : 0);
    printf("\n");

    /* Open output file — fd for parallel pwrite, FILE* for header/metadata */
    int out_fd = open(output_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        fprintf(stderr, "Cannot create: %s\n", output_path);
        free(out_types); free(out_sizes); free(out_offsets);
        ct_gguf_close(src);
        return 1;
    }
    FILE* out = fdopen(out_fd, "wb");
    if (!out) {
        fprintf(stderr, "Cannot fdopen: %s\n", output_path);
        close(out_fd); remove(output_path);
        free(out_types); free(out_sizes); free(out_offsets);
        ct_gguf_close(src);
        return 1;
    }

    /* === PASS 2a: Write header + metadata + tensor info === */
    printf("=== Writing header... ===\n");
    if (!write_header_and_metadata(out, src, out_types, out_sizes, out_offsets, n)) {
        fprintf(stderr, "Failed to write header\n");
        fclose(out); remove(output_path);
        free(out_types); free(out_sizes); free(out_offsets);
        ct_gguf_close(src);
        return 1;
    }
    fflush(out);  /* flush header before threads start writing */

    /* === PASS 2b: Parallel tensor conversion === */
    long data_start = ftell(out);
    printf("=== Parallel conversion (%d tensors, %d threads) ===\n", n, n > 4 ? 4 : n);

    /* Determine thread count — use up to 4 threads for ARM big.LITTLE */
    int n_threads = n < 4 ? n : 4;
    if (n < n_threads) n_threads = n;

    pthread_t threads[4];
    worker_arg args[4];
    volatile bool global_ok = true;

    int chunk = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        args[t].thread_id = t;
        args[t].tensor_start = t * chunk;
        args[t].tensor_end = (t == n_threads - 1) ? n : (t + 1) * chunk;
        args[t].src = src;
        args[t].out_fd = out_fd;
        args[t].data_start = (size_t)data_start;
        args[t].out_offsets = out_offsets;
        args[t].out_types = out_types;
        args[t].dst_type = dst_type;
        args[t].calibrate = calibrate;
        args[t].format_name = format_name;
        args[t].preserve_head = preserve_sensitive;
        args[t].global_ok = &global_ok;
        pthread_create(&threads[t], NULL, worker_convert, &args[t]);
    }

    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    bool ok = global_ok;

    /* Cleanup */
    fclose(out);  /* also closes out_fd via fdopen */
    free(out_types);
    free(out_sizes);
    free(out_offsets);
    ct_gguf_close(src);

    if (ok) {
        printf("\n  ✅ Done! Output: %s (%.2f GB)\n",
               output_path, total_output / (1024.0*1024.0*1024.0));
        if (verify) {
            printf("\n=== Verification ===\n");
            if (!verify_gguf(output_path))
                return 1;
        }
        return 0;
    } else {
        fprintf(stderr, "\n  ❌ Failed during conversion\n");
        return 1;
    }
}

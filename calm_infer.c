/**
 * calm_infer.c — Calm Native Inference Engine
 *
 * LLaMA-family transformer full forward pass + sampling.
 * Weight dispatch: F32 / F16 / Q8_0 / Q4_0 / BQ1_0 / TQ1_0.
 *
 * Pipeline per token:
 *   embed → [RMS norm → QKV proj → RoPE → cache → attention → output proj → +residual] × N
 *        → [RMS norm → FFN gate/up → SiLU → FFN down → +residual] × N
 *        → final RMS norm → output proj → logits → sample
 */
#include "calm_infer.h"
#include "calm_quant.h"
#include "calm_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef CT_VULKAN
#include "calm_vulkan.h"
/* Weight lookup table: maps CPU pointer → Vulkan weight ID */
typedef struct { const void* cpu_ptr; ct_vulkan_weight_id wid; int I, O; } vk_we;
static vk_we g_vk_w[1024];
static int g_vk_n = 0;
static ct_vulkan_backend* g_vk = NULL;
static int g_vk_ready = 0;

static void vk_add(const void* p, ct_vulkan_weight_id wid, int I, int O) {
    if (g_vk_n < 1024) { g_vk_w[g_vk_n].cpu_ptr = p; g_vk_w[g_vk_n].wid = wid; g_vk_w[g_vk_n].I = I; g_vk_w[g_vk_n].O = O; g_vk_n++; }
}
static ct_vulkan_weight_id vk_find(const void* p, int* I, int* O) {
    for (int i = 0; i < g_vk_n; i++) { if (g_vk_w[i].cpu_ptr == p) { *I = g_vk_w[i].I; *O = g_vk_w[i].O; return g_vk_w[i].wid; } }
    return -1;
}
/* Called from ct_infer_create — initializes Vulkan once */
static void vk_init_backend(void) {
    if (g_vk_ready) return;
    g_vk_ready = 1;
    fprintf(stderr, "vk: initializing backend...\n");
    g_vk = ct_vulkan_init();
    if (g_vk) fprintf(stderr, "vk: backend ready (%s)\n", ct_vulkan_device_name(g_vk));
    else fprintf(stderr, "vk: init failed, using CPU\n");
}
/* Upload one Q8_0 weight tensor to Vulkan, storing lookup entry */
static void vk_upload_weight(const void* w, int I, int O, const char* name) {
    if (!g_vk) return;
    ct_vulkan_weight_id wid = ct_vulkan_upload_weights(g_vk, CT_GGUF_TYPE_Q8_0, w, I, O, name);
    if (wid >= 0) { vk_add(w, wid, I, O); }
}
#endif /* CT_VULKAN */

/* ═══════════════════════════════════════════════════════════════
 * K-quant block structures (GGUF v3 spec)
 * ═══════════════════════════════════════════════════════════════ */

#define CT_QK_K 256

#pragma pack(push, 1)
typedef struct {
    uint16_t d;     /* FP16 scale */
    uint8_t  qh[4]; /* 5th bit for 32 values */
    uint8_t  qs[16];/* Low 4 bits for 32 values */
} ct_block_q5_0;

typedef struct {
    uint16_t d;       /* FP16 super-block scale */
    uint16_t dmin;    /* FP16 super-block min */
    uint8_t  scales[8];  /* Sub-block 6-bit scales */
    uint8_t  qs[128];    /* 4-bit quants for 256 values */
} ct_block_q4_K;

typedef struct {
    uint16_t d;        /* FP16 super-block scale */
    uint8_t  ql[128];  /* Lower 4 bits of quants */
    uint8_t  qh[64];   /* Upper 2 bits of quants (packed) */
    int8_t   scales[16]; /* 8-bit sub-block scales */
} ct_block_q6_K;

typedef struct {
    uint16_t d;       /* FP16 super-block scale */
    uint8_t  qs[128]; /* 8-bit quants for 128 values */
} ct_block_q8_K;
#pragma pack(pop)

/* Dequantize one Q5_0 block of 32 values to float */
static void deq_q5_0(const ct_block_q5_0* b, float* out) {
    float d = ct_fp16_to_fp32(b->d);
    for (int i = 0; i < 32; i++) {
        int nib = (b->qs[i >> 1] >> ((i & 1) << 2)) & 0xF;
        int hi  = (b->qh[i >> 3] >> (i & 7)) & 1;
        out[i] = ((float)(nib | (hi << 4)) - 16.0f) * d;
    }
}

/* Dequantize one Q4_K super-block of 256 values to float */
static void deq_q4_K(const ct_block_q4_K* b, float* out) {
    float d    = ct_fp16_to_fp32(b->d);
    float dmin = ct_fp16_to_fp32(b->dmin);
    for (int j = 0; j < 256; j++) {
        int iscale = j / 32;
        int sc = (b->scales[iscale / 2] >> ((iscale & 1) << 2)) & 0xF;
        sc = (sc & 8) ? (sc | 0xF0) : sc; /* sign-extend 4-bit */
        float s  = d * sc;
        float sm = dmin * (sc - 0x10);
        int nib  = (b->qs[j >> 1] >> ((j & 1) << 2)) & 0xF;
        out[j] = (float)nib * s + sm;
    }
}

/* Dequantize one Q6_K super-block of 256 values to float */
static void deq_q6_K(const ct_block_q6_K* b, float* out) {
    float d = ct_fp16_to_fp32(b->d);
    for (int j = 0; j < 256; j++) {
        int ql = (b->ql[j >> 1] >> ((j & 1) << 2)) & 0xF;
        int qh = (b->qh[j >> 2] >> ((j & 3) << 1)) & 3;
        int val = ql | (qh << 4);
        if (val > 63) val -= 128; /* sign-extend 7-bit */
        int sc_idx = j / 16;
        out[j] = (float)val * (float)b->scales[sc_idx] * d;
    }
}

/* Dequantize one Q8_K super-block of 128 values to float */
static void deq_q8_K(const ct_block_q8_K* b, float* out) {
    float d = ct_fp16_to_fp32(b->d);
    for (int j = 0; j < 128; j++)
        out[j] = (float)((int8_t)b->qs[j]) * d;
}

/* Scalar K-quant matmul — dequant block + dot product */

static void matmul_q5_0(float* y, const float* x, const ct_block_q5_0* w, int I, int O) {
    float buf[32];
    for (int o = 0; o < O; o++) {
        float sum = 0.0f;
        int base = o * I;
        for (int b = 0; b < I / 32; b++) {
            deq_q5_0(&w[base / 32 + b], buf);
            for (int i = 0; i < 32; i++)
                sum += x[b * 32 + i] * buf[i];
        }
        y[o] = sum;
    }
}

static void matmul_q4_K(float* y, const float* x, const ct_block_q4_K* w, int I, int O) {
    float buf[256];
    for (int o = 0; o < O; o++) {
        float sum = 0.0f;
        int base = o * I;
        for (int b = 0; b < I / CT_QK_K; b++) {
            deq_q4_K(&w[base / CT_QK_K + b], buf);
            for (int i = 0; i < CT_QK_K; i++)
                sum += x[b * CT_QK_K + i] * buf[i];
        }
        y[o] = sum;
    }
}

static void matmul_q6_K(float* y, const float* x, const ct_block_q6_K* w, int I, int O) {
    float buf[256];
    for (int o = 0; o < O; o++) {
        float sum = 0.0f;
        int base = o * I;
        for (int b = 0; b < I / CT_QK_K; b++) {
            deq_q6_K(&w[base / CT_QK_K + b], buf);
            for (int i = 0; i < CT_QK_K; i++)
                sum += x[b * CT_QK_K + i] * buf[i];
        }
        y[o] = sum;
    }
}

static void matmul_q8_K(float* y, const float* x, const ct_block_q8_K* w, int I, int O) {
    float buf[128];
    for (int o = 0; o < O; o++) {
        float sum = 0.0f;
        int base = o * I;
        for (int b = 0; b < I / 128; b++) {
            deq_q8_K(&w[base / 128 + b], buf);
            for (int i = 0; i < 128; i++)
                sum += x[b * 128 + i] * buf[i];
        }
        y[o] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * F16 matmul (scalar — dequantize on the fly)
 * ═══════════════════════════════════════════════════════════════ */

static void matmul_f16(float* y, const float* x, const uint16_t* W, int I, int O) {
    for (int i = 0; i < O; i++) {
        float sum = 0.0f;
        int base = i * I;
        for (int j = 0; j < I; j++)
            sum += x[j] * ct_fp16_to_fp32(W[base + j]);
        y[i] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Matmul dispatch: y[O] = x[I] @ weights[O, I]
 * ═══════════════════════════════════════════════════════════════ */

void matmul(float* y, const float* x, const void* w, int type, int I, int O) {
#ifdef CT_VULKAN
    /* Try Vulkan Q8_0 dispatch first */
    if (type == CT_GGUF_TYPE_Q8_0 && g_vk && ct_vulkan_available(g_vk)) {
        int vk_I, vk_O;
        ct_vulkan_weight_id wid = vk_find(w, &vk_I, &vk_O);
        if (wid >= 0 && vk_I == I && vk_O == O) {
            if (ct_vulkan_batch_active(g_vk)) {
                /* Use batch path when between batch_begin/batch_end */
                if (ct_vulkan_batch_matmul_q8_0(g_vk, wid, x, y, I, O) == 0)
                    return;
            } else {
                if (ct_vulkan_matmul_q8_0(g_vk, wid, x, y, I, O) == 0)
                    return;
            }
        }
        /* Vulkan dispatch failed — fall through to CPU */
    }
#endif
    switch (type) {
        case CT_GGUF_TYPE_F32:
            ct_matmul_f32(y, x, (const float*)w, I, O);
            break;
        case CT_GGUF_TYPE_F16:
            matmul_f16(y, x, (const uint16_t*)w, I, O);
            break;
        case CT_GGUF_TYPE_Q8_0:
            ct_matmul_q8_0(y, x, (const ct_block_q8_0*)w, I, O);
            break;
        case CT_GGUF_TYPE_Q4_0:
            ct_matmul_q4_0(y, x, (const ct_block_q4_0*)w, I, O);
            break;
        case CT_GGUF_TYPE_BQ1_0:
            ct_matmul_bq1_0(y, x, (const ct_block_bq1_0*)w, I, O);
            break;
        case CT_GGUF_TYPE_TQ1_0:
            ct_matmul_tq1_0(y, x, (const ct_block_tq1_0*)w, I, O);
            break;
        case CT_GGUF_TYPE_Q5_0:
            matmul_q5_0(y, x, (const ct_block_q5_0*)w, I, O);
            break;
        case CT_GGUF_TYPE_Q4_K:
            matmul_q4_K(y, x, (const ct_block_q4_K*)w, I, O);
            break;
        case CT_GGUF_TYPE_Q6_K:
            matmul_q6_K(y, x, (const ct_block_q6_K*)w, I, O);
            break;
        case CT_GGUF_TYPE_Q8_K:
            matmul_q8_K(y, x, (const ct_block_q8_K*)w, I, O);
            break;
        default:
            fprintf(stderr, "infer: unsupported matmul type %d (I=%d O=%d)\n", type, I, O);
            memset(y, 0, (size_t)O * sizeof(float));
            break;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * RMS Normalization: y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
 * ═══════════════════════════════════════════════════════════════ */

void rms_norm(float* y, const float* x, const float* w,
              int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        y[i] = x[i] * scale * w[i];
}

/* ═══════════════════════════════════════════════════════════════
 * SiLU (Sigmoid Linear Unit): silu(x) = x / (1 + exp(-x))
 * ═══════════════════════════════════════════════════════════════ */

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ═══════════════════════════════════════════════════════════════
 * RoPE — Rotary Position Embedding (NeurIPS 2021)
 *
 * Applies rotation to pairs (x[2i], x[2i+1]) by angle p * theta_i
 * where theta_i = 1 / base^(2i/d)
 *
 * In-place on buf[n] where n must be even.
 * ═══════════════════════════════════════════════════════════════ */

void rope(float* buf, int n, int pos, float base) {
    float theta = 1.0f;
    float theta_scale = powf(base, -2.0f / (float)n);
    for (int i = 0; i < n; i += 2) {
        float p_theta = (float)pos * theta;
        float c = cosf(p_theta);
        float s = sinf(p_theta);
        float x0 = buf[i];
        float x1 = buf[i + 1];
        buf[i]     = x0 * c - x1 * s;
        buf[i + 1] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Embedding lookup: copy one row from the embedding table.
 * Handles F32, Q8_0, Q4_0 (most models use F32 for embeddings).
 * ═══════════════════════════════════════════════════════════════ */

void embed_row(float* out, const void* table, int type,
               int token, int n_embd) {
    /* Compute byte offset for the requested token's row */
    size_t row_size;
    uint64_t dims[2] = {(uint64_t)n_embd, 1};
    row_size = ct_gguf_tensor_size(type, 2, dims);
    const uint8_t* row = (const uint8_t*)table + (size_t)token * row_size;

    switch (type) {
        case CT_GGUF_TYPE_F32:
            memcpy(out, row, (size_t)n_embd * sizeof(float));
            break;
        case CT_GGUF_TYPE_F16: {
            const uint16_t* f16 = (const uint16_t*)row;
            for (int i = 0; i < n_embd; i++)
                out[i] = ct_fp16_to_fp32(f16[i]);
            break;
        }
        case CT_GGUF_TYPE_Q8_0: {
            const ct_block_q8_0* blocks = (const ct_block_q8_0*)row;
            int nb = (n_embd + 31) / 32;
            for (int b = 0; b < nb; b++) {
                float d = ct_fp16_to_fp32(blocks[b].d);
                int rem = n_embd - b * 32;
                int n = rem < 32 ? rem : 32;
                for (int i = 0; i < n; i++)
                    out[b * 32 + i] = blocks[b].qs[i] * d;
            }
            break;
        }
        case CT_GGUF_TYPE_Q4_0: {
            const ct_block_q4_0* blocks = (const ct_block_q4_0*)row;
            int nb = (n_embd + 31) / 32;
            for (int b = 0; b < nb; b++) {
                float d = ct_fp16_to_fp32(blocks[b].d);
                int rem = n_embd - b * 32;
                int n = rem < 32 ? rem : 32;
                for (int i = 0; i < n; i++) {
                    int nib = (blocks[b].qs[i >> 1] >> ((i & 1) << 2)) & 0xF;
                    out[b * 32 + i] = ((float)nib - 8.0f) * d;
                }
            }
            break;
        }
        case CT_GGUF_TYPE_Q5_0: {
            const ct_block_q5_0* blocks = (const ct_block_q5_0*)row;
            int nb = (n_embd + 31) / 32;
            for (int b = 0; b < nb; b++)
                deq_q5_0(&blocks[b], out + b * 32);
            break;
        }
        case CT_GGUF_TYPE_Q4_K: {
            const ct_block_q4_K* blocks = (const ct_block_q4_K*)row;
            int nb = (n_embd + 255) / 256;
            for (int b = 0; b < nb; b++)
                deq_q4_K(&blocks[b], out + b * 256);
            break;
        }
        case CT_GGUF_TYPE_Q6_K: {
            const ct_block_q6_K* blocks = (const ct_block_q6_K*)row;
            int nb = (n_embd + 255) / 256;
            for (int b = 0; b < nb; b++)
                deq_q6_K(&blocks[b], out + b * 256);
            break;
        }
        case CT_GGUF_TYPE_Q8_K: {
            const ct_block_q8_K* blocks = (const ct_block_q8_K*)row;
            int nb = (n_embd + 127) / 128;
            for (int b = 0; b < nb; b++)
                deq_q8_K(&blocks[b], out + b * 128);
            break;
        }
        default:
            fprintf(stderr, "infer: unsupported embed type %d\n", type);
            memset(out, 0, (size_t)n_embd * sizeof(float));
            break;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Config extraction from GGUF metadata
 *
 * Reads LLaMA-family architecture parameters.
 * Tries both prefixed (llama.X) and unprefixed (X) keys.
 * ═══════════════════════════════════════════════════════════════ */

static uint64_t meta_get(ct_gguf_context* ctx, const char* arch,
                          const char* key, uint64_t def) {
    char full[256];
    /* Try: arch.key */
    snprintf(full, sizeof(full), "%s.%s", arch, key);
    uint64_t v = ct_gguf_meta_uint(ctx, full, UINT64_MAX);
    if (v != UINT64_MAX) return v;
    /* Try: key only */
    v = ct_gguf_meta_uint(ctx, key, UINT64_MAX);
    if (v != UINT64_MAX) return v;
    /* Try: tokenizer.ggml.key (for vocab_size) */
    snprintf(full, sizeof(full), "tokenizer.ggml.%s", key);
    v = ct_gguf_meta_uint(ctx, full, UINT64_MAX);
    if (v != UINT64_MAX) return v;
    return def;
}

static float meta_get_float(ct_gguf_context* ctx, const char* arch,
                             const char* key, float def) {
    char full[256];
    snprintf(full, sizeof(full), "%s.%s", arch, key);
    const char* s = ct_gguf_meta_string(ctx, full);
    if (s) return (float)atof(s);
    /* Try as float32/float64 directly from metadata */
    for (int i = 0; i < ctx->metadata.count; i++) {
        if (strcmp(ctx->metadata.keys[i], full) == 0) {
            switch (ctx->metadata.types[i]) {
                case CT_GGUF_VALUE_FLOAT32: return ctx->metadata.values[i].v_f32;
                case CT_GGUF_VALUE_FLOAT64: return (float)ctx->metadata.values[i].v_f64;
            }
        }
    }
    /* Try as uint */
    uint64_t v = ct_gguf_meta_uint(ctx, full, UINT64_MAX);
    if (v != UINT64_MAX) return (float)v;
    /* Try without prefix */
    s = ct_gguf_meta_string(ctx, key);
    if (s) return (float)atof(s);
    for (int i = 0; i < ctx->metadata.count; i++) {
        if (strcmp(ctx->metadata.keys[i], key) == 0) {
            switch (ctx->metadata.types[i]) {
                case CT_GGUF_VALUE_FLOAT32: return ctx->metadata.values[i].v_f32;
                case CT_GGUF_VALUE_FLOAT64: return (float)ctx->metadata.values[i].v_f64;
            }
        }
    }
    v = ct_gguf_meta_uint(ctx, key, UINT64_MAX);
    if (v != UINT64_MAX) return (float)v;
    return def;
}

static int extract_config(ct_gguf_context* gguf, ct_infer_config* cfg) {
    const char* arch = ct_gguf_architecture(gguf);
    if (!arch) arch = "unknown";

    cfg->n_layer    = (int)meta_get(gguf, arch, "block_count", 0);
    cfg->n_embd     = (int)meta_get(gguf, arch, "embedding_length", 0);
    cfg->n_head     = (int)meta_get(gguf, arch, "attention.head_count", 0);
    cfg->n_head_kv  = (int)meta_get(gguf, arch, "attention.head_count_kv", cfg->n_head);
    cfg->n_ff       = (int)meta_get(gguf, arch, "feed_forward_length", 0);
    cfg->n_ctx_max  = (int)meta_get(gguf, arch, "context_length", 2048);
    cfg->n_vocab    = (int)meta_get(gguf, arch, "vocab_size", 0);
    cfg->head_dim   = (int)meta_get(gguf, arch, "attention.head_dim", 0);
    cfg->norm_rms_eps = meta_get_float(gguf, arch,
                                        "attention.layer_norm_rms_epsilon", 1e-6f);
    cfg->rope_freq_base = meta_get_float(gguf, arch, "rope.freq_base", 10000.0f);

    /* MoE config (defaults to 0 = dense model) */
    cfg->n_expert          = (int)meta_get(gguf, arch, "expert_count", 0);
    cfg->n_expert_per_token = (int)meta_get(gguf, arch, "expert_used_count", 0);

    /* SSM (Mamba) config — defaults to 0 (not an SSM model) */
    cfg->ssm_d_conv     = (int)meta_get(gguf, arch, "ssm.conv_kernel", 0);
    cfg->ssm_d_inner    = (int)meta_get(gguf, arch, "ssm.inner_size", 0);
    cfg->ssm_d_state    = (int)meta_get(gguf, arch, "ssm.state_size", 0);
    cfg->ssm_dt_rank    = (int)meta_get(gguf, arch, "ssm.time_step_rank", 0);
    cfg->ssm_dt_b_c_rms = (int)meta_get(gguf, arch, "ssm.dt_b_c_rms", 0);

    /* If ssm_d_inner not stored, default to 2 * n_embd (Mamba convention) */
    if (cfg->ssm_d_conv > 0 && cfg->ssm_d_inner == 0)
        cfg->ssm_d_inner = 2 * cfg->n_embd;

    /* If head_dim not explicitly stored, infer from n_embd / n_head */
    if (cfg->head_dim == 0 && cfg->n_head > 0)
        cfg->head_dim = cfg->n_embd / cfg->n_head;
    /* If n_ff not stored, infer as 8/3 * n_embd (typical LLaMA) */
    if (cfg->n_ff == 0)
        cfg->n_ff = cfg->n_embd * 8 / 3;

    return (cfg->n_layer > 0 && cfg->n_embd > 0 && cfg->n_head > 0) ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════
 * Weight table construction
 *
 * Scans all tensors in the GGUF context and stores pointers
 * for each known weight name.
 * ═══════════════════════════════════════════════════════════════ */

/* Find tensor by exact name match. Returns type and data pointer. */
static const void* find_tensor(ct_gguf_context* gguf, const char* name,
                                int* out_type) {
    const ct_gguf_tensor_info* t = ct_gguf_find_tensor(gguf, name);
    if (!t) return NULL;
    *out_type = t->type;
    return ct_gguf_tensor_data(gguf, t);
}

static int build_weights(ct_gguf_context* gguf, ct_infer_weights* w) {
    const int L = w->config.n_layer;

    /* Token embeddings */
    w->token_embd = (void*)find_tensor(gguf, "token_embd.weight", &w->t_embd);
    if (!w->token_embd) {
        fprintf(stderr, "infer: missing token_embd.weight\n");
    } else {
        fprintf(stderr, "infer: token_embd.weight type=%d\n", w->t_embd);
    }

    /* Final norm */
    {
        int tmp_type = 0;
        w->final_norm = (float*)find_tensor(gguf, "output_norm.weight", &tmp_type);
        fprintf(stderr, "infer: output_norm.weight type=%d\n", tmp_type);
        if (!w->final_norm) {
            /* Some models don't have an explicit final norm name.
             * Try to use the last layer's FFN norm instead. */
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", L - 1);
            int t3 = 0;
            w->final_norm = (float*)find_tensor(gguf, name, &t3);
        }
        if (!w->final_norm) {
            fprintf(stderr, "infer: missing output_norm.weight\n");
            return -1;
        }
    }

    /* Output projection */
    w->output_weight = (void*)find_tensor(gguf, "output.weight", &w->t_out);
    if (w->output_weight)
        fprintf(stderr, "infer: output.weight type=%d\n", w->t_out);
    else
        fprintf(stderr, "infer: no output.weight, using token_embd for LM head\n");

    /* Allocate layer array */
    w->layers = (ct_infer_layer*)calloc((size_t)L, sizeof(ct_infer_layer));
    if (!w->layers) return -1;

    for (int i = 0; i < L; i++) {
        ct_infer_layer* l = &w->layers[i];
        char name[128];
        int tmp_type;

        /* Detect layer type: check for SSM vs Attention */
        snprintf(name, sizeof(name), "blk.%d.ssm_in.weight", i);
        l->is_ssm = (find_tensor(gguf, name, &tmp_type) != NULL) ? 1 : 0;

        /* Load attn_norm (used by both attention and SSM blocks) */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        l->attn_norm = (float*)find_tensor(gguf, name, &tmp_type);
        if (i == 0) fprintf(stderr, "infer: attn_norm type=%d (layer 0)\n", tmp_type);
        if (!l->attn_norm) {
            fprintf(stderr, "infer: missing blk.%d.attn_norm.weight\n", i);
            return -1;
        }

        if (l->is_ssm) {
            /* ── SSM (Mamba) layer — no attention weights ── */
            if (i == 0) fprintf(stderr, "infer: blk.0 is SSM layer (ssm_in)\n");

            snprintf(name, sizeof(name), "blk.%d.ssm_in.weight", i);
            l->ssm_in = (void*)find_tensor(gguf, name, &l->t_ssm_in);
            if (!l->ssm_in) { fprintf(stderr, "infer: missing %s\n", name); return -1; }

            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
            l->ssm_conv1d = (void*)find_tensor(gguf, name, &l->t_ssm_conv1d);
            if (!l->ssm_conv1d) { fprintf(stderr, "infer: missing %s\n", name); return -1; }

            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.bias", i);
            l->ssm_conv1d_b = (float*)find_tensor(gguf, name, &tmp_type);

            snprintf(name, sizeof(name), "blk.%d.ssm_x.weight", i);
            l->ssm_x = (void*)find_tensor(gguf, name, &l->t_ssm_x);
            if (!l->ssm_x) { fprintf(stderr, "infer: missing %s\n", name); return -1; }

            /* Optional RMS norm weights for dt, B, C (Jamba-style) */
            snprintf(name, sizeof(name), "blk.%d.ssm_dt_norm.weight", i);
            l->ssm_dt_norm = (float*)find_tensor(gguf, name, &tmp_type);
            snprintf(name, sizeof(name), "blk.%d.ssm_b_norm.weight", i);
            l->ssm_b_norm = (float*)find_tensor(gguf, name, &tmp_type);
            snprintf(name, sizeof(name), "blk.%d.ssm_c_norm.weight", i);
            l->ssm_c_norm = (float*)find_tensor(gguf, name, &tmp_type);

            snprintf(name, sizeof(name), "blk.%d.ssm_dt.weight", i);
            l->ssm_dt = (void*)find_tensor(gguf, name, &l->t_ssm_dt);
            if (!l->ssm_dt) { fprintf(stderr, "infer: missing %s\n", name); return -1; }

            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", i);
            l->ssm_dt_b = (float*)find_tensor(gguf, name, &tmp_type);

            snprintf(name, sizeof(name), "blk.%d.ssm_a", i);
            l->ssm_a = (void*)find_tensor(gguf, name, &l->t_ssm_a);
            if (!l->ssm_a) { fprintf(stderr, "infer: missing %s\n", name); return -1; }

            snprintf(name, sizeof(name), "blk.%d.ssm_d", i);
            l->ssm_d = (void*)find_tensor(gguf, name, &l->t_ssm_d);
            if (!l->ssm_d) { fprintf(stderr, "infer: missing %s\n", name); return -1; }

            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", i);
            l->ssm_out = (void*)find_tensor(gguf, name, &l->t_ssm_out);
            if (!l->ssm_out) { fprintf(stderr, "infer: missing %s\n", name); return -1; }

            /* Zero out attention pointers (safety) */
            l->attn_q = l->attn_k = l->attn_v = NULL;
            l->attn_out = NULL;
            l->attn_q_bias = l->attn_k_bias = l->attn_v_bias = NULL;

            /* Print SSM layer summary once */
            if (i == 0) {
                fprintf(stderr, "infer: ssm_in type=%d, ssm_conv1d type=%d, ssm_x type=%d, "
                        "ssm_dt type=%d, ssm_a type=%d, ssm_d type=%d, ssm_out type=%d\n",
                        l->t_ssm_in, l->t_ssm_conv1d, l->t_ssm_x,
                        l->t_ssm_dt, l->t_ssm_a, l->t_ssm_d, l->t_ssm_out);
            }
        } else {
            /* ── Attention layer (original path) ── */
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            l->attn_q = (void*)find_tensor(gguf, name, &l->t_q);
            if (!l->attn_q) {
                fprintf(stderr, "infer: missing blk.%d.attn_q.weight\n", i);
                return -1;
            }
            if (i == 0) fprintf(stderr, "infer: blk.0.attn_q.weight type=%d\n", l->t_q);
            snprintf(name, sizeof(name), "blk.%d.attn_q.bias", i);
            l->attn_q_bias = (float*)find_tensor(gguf, name, &tmp_type);
            if (i == 0 && l->attn_q_bias) fprintf(stderr, "infer: attn_q.bias type=%d (layer 0)\n", tmp_type);

            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
            l->attn_k = (void*)find_tensor(gguf, name, &l->t_k);
            if (!l->attn_k) {
                fprintf(stderr, "infer: missing blk.%d.attn_k.weight\n", i);
                return -1;
            }
            if (i == 0) fprintf(stderr, "infer: blk.0.attn_k.weight type=%d\n", l->t_k);
            snprintf(name, sizeof(name), "blk.%d.attn_k.bias", i);
            l->attn_k_bias = (float*)find_tensor(gguf, name, &tmp_type);

            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
            l->attn_v = (void*)find_tensor(gguf, name, &l->t_v);
            if (!l->attn_v) {
                fprintf(stderr, "infer: missing blk.%d.attn_v.weight\n", i);
                return -1;
            }
            if (i == 0) fprintf(stderr, "infer: blk.0.attn_v.weight type=%d\n", l->t_v);
            snprintf(name, sizeof(name), "blk.%d.attn_v.bias", i);
            l->attn_v_bias = (float*)find_tensor(gguf, name, &tmp_type);

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
            l->attn_out = (void*)find_tensor(gguf, name, &l->t_o);
            if (!l->attn_out) {
                fprintf(stderr, "infer: missing blk.%d.attn_output.weight\n", i);
                return -1;
            }
            if (i == 0) fprintf(stderr, "infer: blk.0.attn_output.weight type=%d\n", l->t_o);
        }

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        l->ffn_norm = (float*)find_tensor(gguf, name, &tmp_type);
        if (!l->ffn_norm) {
            fprintf(stderr, "infer: missing blk.%d.ffn_norm.weight\n", i);
            return -1;
        }

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        l->ffn_gate = (void*)find_tensor(gguf, name, &l->t_g);
        if (!l->ffn_gate) {
            fprintf(stderr, "infer: missing blk.%d.ffn_gate.weight\n", i);
            return -1;
        }
        if (i == 0) fprintf(stderr, "infer: blk.0.ffn_gate.weight type=%d\n", l->t_g);

        /* ffn_up and ffn_down are optional for MoE models (experts have their own) */
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        l->ffn_up = (void*)find_tensor(gguf, name, &l->t_u);
        if (!l->ffn_up) {
            if (w->config.n_expert == 0) {
                fprintf(stderr, "infer: missing blk.%d.ffn_up.weight\n", i);
                return -1;
            }
            l->t_u = 0;
        }
        if (i == 0 && l->ffn_up)
            fprintf(stderr, "infer: blk.0.ffn_up.weight type=%d\n", l->t_u);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        l->ffn_down = (void*)find_tensor(gguf, name, &l->t_d);
        if (!l->ffn_down) {
            if (w->config.n_expert == 0) {
                fprintf(stderr, "infer: missing blk.%d.ffn_down.weight\n", i);
                return -1;
            }
            l->t_d = 0;
        }
        if (i == 0 && l->ffn_down)
            fprintf(stderr, "infer: blk.0.ffn_down.weight type=%d\n", l->t_d);

        /* MoE expert weights (if applicable) */
        if (w->config.n_expert > 0) {
            int ne = w->config.n_expert;
            l->expert_gate = (void**)calloc((size_t)ne, sizeof(void*));
            l->expert_up   = (void**)calloc((size_t)ne, sizeof(void*));
            l->expert_down = (void**)calloc((size_t)ne, sizeof(void*));
            l->t_eg = (int*)calloc((size_t)ne, sizeof(int));
            l->t_eu = (int*)calloc((size_t)ne, sizeof(int));
            l->t_ed = (int*)calloc((size_t)ne, sizeof(int));
            if (!l->expert_gate || !l->expert_up || !l->expert_down ||
                !l->t_eg || !l->t_eu || !l->t_ed) return -1;

            for (int e = 0; e < ne; e++) {
                snprintf(name, sizeof(name), "blk.%d.experts.%d.ffn_gate.weight", i, e);
                l->expert_gate[e] = (void*)find_tensor(gguf, name, &l->t_eg[e]);
                if (!l->expert_gate[e]) {
                    fprintf(stderr, "infer: missing %s\n", name);
                    return -1;
                }
                snprintf(name, sizeof(name), "blk.%d.experts.%d.ffn_up.weight", i, e);
                l->expert_up[e] = (void*)find_tensor(gguf, name, &l->t_eu[e]);
                if (!l->expert_up[e]) {
                    fprintf(stderr, "infer: missing %s\n", name);
                    return -1;
                }
                snprintf(name, sizeof(name), "blk.%d.experts.%d.ffn_down.weight", i, e);
                l->expert_down[e] = (void*)find_tensor(gguf, name, &l->t_ed[e]);
                if (!l->expert_down[e]) {
                    fprintf(stderr, "infer: missing %s\n", name);
                    return -1;
                }
            }
            if (i == 0)
                fprintf(stderr, "infer: MoE with %d experts, top-%d per token\n",
                        ne, w->config.n_expert_per_token);
        }
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_infer_create
 * ═══════════════════════════════════════════════════════════════ */

ct_infer_state* ct_infer_create(ct_gguf_context* gguf, int max_ctx, int gpu_layers) {
    if (!gguf) return NULL;

    /* Initialize FP16 lookup table for quantized operations */
    ct_quant_init();

    ct_infer_state* s = (ct_infer_state*)calloc(1, sizeof(ct_infer_state));
    if (!s) return NULL;
    s->gguf = gguf;
    s->gpu_layers = (gpu_layers < 0) ? 0 : gpu_layers;

    /* Extract config */
    if (extract_config(gguf, &s->w.config) != 0) {
        fprintf(stderr, "infer: failed to extract model config\n");
        goto fail;
    }
    ct_infer_config* cfg = &s->w.config;

    /* Validate */
    if (cfg->n_layer < 1 || cfg->n_embd < 1 || cfg->n_head < 1 || cfg->head_dim < 1) {
        fprintf(stderr, "infer: invalid config (L=%d E=%d H=%d D=%d)\n",
                cfg->n_layer, cfg->n_embd, cfg->n_head, cfg->head_dim);
        goto fail;
    }

    /* Build weight table */
    if (build_weights(gguf, &s->w) != 0)
        goto fail;

#ifdef CT_VULKAN
    /* Initialize Vulkan backend and upload Q8_0 weights */
    vk_init_backend();
    if (g_vk) {
        fprintf(stderr, "vk: uploading weights...\n");
        ct_infer_weights* wgt = &s->w;
        int n_uploaded = 0;
        /* Upload token_embd if Q8_0 */
        {
            const ct_gguf_tensor_info* t = ct_gguf_find_tensor(gguf, "token_embd.weight");
            if (t && t->type == CT_GGUF_TYPE_Q8_0) {
                int I = (int)t->dims[0], O = (int)t->dims[1];
                vk_upload_weight(ct_gguf_tensor_data(gguf, t), I, O, "token_embd.weight");
            }
        }
        /* Upload output_weight if Q8_0 */
        {
            const ct_gguf_tensor_info* t = ct_gguf_find_tensor(gguf, "output.weight");
            if (!t) t = ct_gguf_find_tensor(gguf, "token_embd.weight");
            if (t && t->type == CT_GGUF_TYPE_Q8_0) {
                const void* data = ct_gguf_tensor_data(gguf, t);
                /* output.weight is a separate tensor, not same as token_embd */
                t = ct_gguf_find_tensor(gguf, "output.weight");
                if (t && t->type == CT_GGUF_TYPE_Q8_0) {
                    int I = (int)t->dims[0], O = (int)t->dims[1];
                    vk_upload_weight(ct_gguf_tensor_data(gguf, t), I, O, "output.weight");
                }
            }
        }
        /* Per-layer Q8_0 weights (only first gpu_layers) */
        int vk_max_layer = (s->gpu_layers >= cfg->n_layer || s->gpu_layers >= 99)
                           ? cfg->n_layer : s->gpu_layers;
        for (int i = 0; i < vk_max_layer; i++) {
            ct_infer_layer* lw = &s->w.layers[i];
            char name[128];
            /* Check each weight in the layer */
            if (lw->t_q == CT_GGUF_TYPE_Q8_0 && lw->attn_q) {
                snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
                vk_upload_weight(lw->attn_q, cfg->n_embd, cfg->n_head * cfg->head_dim, name);
            }
            if (lw->t_k == CT_GGUF_TYPE_Q8_0 && lw->attn_k) {
                snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
                vk_upload_weight(lw->attn_k, cfg->n_embd, cfg->n_head_kv * cfg->head_dim, name);
            }
            if (lw->t_v == CT_GGUF_TYPE_Q8_0 && lw->attn_v) {
                snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
                vk_upload_weight(lw->attn_v, cfg->n_embd, cfg->n_head_kv * cfg->head_dim, name);
            }
            if (lw->t_o == CT_GGUF_TYPE_Q8_0 && lw->attn_out) {
                snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
                vk_upload_weight(lw->attn_out, cfg->n_head * cfg->head_dim, cfg->n_embd, name);
            }
            if (lw->t_g == CT_GGUF_TYPE_Q8_0 && lw->ffn_gate) {
                if (cfg->n_expert > 0) {
                    snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
                    vk_upload_weight(lw->ffn_gate, cfg->n_embd, cfg->n_expert, name);
                } else {
                    snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
                    vk_upload_weight(lw->ffn_gate, cfg->n_embd, cfg->n_ff, name);
                }
            }
            if (lw->t_u == CT_GGUF_TYPE_Q8_0 && lw->ffn_up) {
                snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
                vk_upload_weight(lw->ffn_up, cfg->n_embd, cfg->n_ff, name);
            }
            if (lw->t_d == CT_GGUF_TYPE_Q8_0 && lw->ffn_down) {
                snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
                vk_upload_weight(lw->ffn_down, cfg->n_ff, cfg->n_embd, name);
            }
            /* MoE expert weights (if any) */
            if (cfg->n_expert > 0 && lw->expert_gate) {
                for (int e = 0; e < cfg->n_expert; e++) {
                    if (lw->t_eg[e] == CT_GGUF_TYPE_Q8_0 && lw->expert_gate[e]) {
                        snprintf(name, sizeof(name), "blk.%d.expert.%d.gate.weight", i, e);
                        vk_upload_weight(lw->expert_gate[e], cfg->n_embd, cfg->n_ff, name);
                    }
                    if (lw->t_eu[e] == CT_GGUF_TYPE_Q8_0 && lw->expert_up[e]) {
                        snprintf(name, sizeof(name), "blk.%d.expert.%d.up.weight", i, e);
                        vk_upload_weight(lw->expert_up[e], cfg->n_embd, cfg->n_ff, name);
                    }
                    if (lw->t_ed[e] == CT_GGUF_TYPE_Q8_0 && lw->expert_down[e]) {
                        snprintf(name, sizeof(name), "blk.%d.expert.%d.down.weight", i, e);
                        vk_upload_weight(lw->expert_down[e], cfg->n_ff, cfg->n_embd, name);
                    }
                }
            }
        }
        fprintf(stderr, "vulkan: uploaded %d weight tensors\n", g_vk_n);
        s->vk_backend = g_vk;
    }
#endif

    /* If n_vocab still 0, infer from token_embd.weight tensor dimensions */
    if (cfg->n_vocab == 0) {
        const ct_gguf_tensor_info* t = ct_gguf_find_tensor(gguf, "token_embd.weight");
        if (t && t->n_dims >= 2) {
            /* Shape is [n_embd, n_vocab] — the second dim is vocab size */
            if (t->dims[0] == (uint64_t)cfg->n_embd)
                cfg->n_vocab = (int)t->dims[1];
            else if (t->dims[1] == (uint64_t)cfg->n_embd)
                cfg->n_vocab = (int)t->dims[0];
            else
                cfg->n_vocab = (int)(t->dims[0] > t->dims[1] ? t->dims[0] : t->dims[1]);
            fprintf(stderr, "infer: inferred vocab_size=%d from token_embd dims (%llu,%llu)\n",
                    cfg->n_vocab, (unsigned long long)t->dims[0], (unsigned long long)t->dims[1]);
        }
    }

    /* KV cache: override context with plan value, cap to reasonable size */
    if (max_ctx > 0 && max_ctx < cfg->n_ctx_max)
        cfg->n_ctx_max = max_ctx;
    if (cfg->n_ctx_max > 4096) cfg->n_ctx_max = 4096;
    s->max_ctx = cfg->n_ctx_max;
    int nhkv = cfg->n_head_kv;
    int hd   = cfg->head_dim;
    size_t kv_elems = (size_t)cfg->n_layer * nhkv * hd * s->max_ctx;

    s->k_cache = (float*)calloc(kv_elems, sizeof(float));
    s->v_cache = (float*)calloc(kv_elems, sizeof(float));

    /* Working buffers — allocate to max needed size across all uses */
    int buf_k_size = cfg->n_ff > cfg->n_embd ? cfg->n_ff : cfg->n_embd;
    if (nhkv * hd > buf_k_size) buf_k_size = nhkv * hd;
    /* SSM d_inner may exceed n_ff, ensure buffer is large enough */
    if (cfg->ssm_d_inner > buf_k_size) buf_k_size = cfg->ssm_d_inner;
    s->hidden  = (float*)calloc((size_t)cfg->n_embd, sizeof(float));
    s->normed  = (float*)calloc((size_t)cfg->n_embd, sizeof(float));
    s->buf_q   = (float*)calloc((size_t)cfg->n_embd, sizeof(float));
    s->buf_k   = (float*)calloc((size_t)buf_k_size, sizeof(float));
    s->buf_v   = (float*)calloc((size_t)cfg->n_embd, sizeof(float));
    s->scores  = (float*)calloc((size_t)s->max_ctx, sizeof(float));
    s->ffbuf   = (float*)calloc((size_t)cfg->n_ff, sizeof(float));
    s->logits  = (float*)calloc((size_t)cfg->n_vocab, sizeof(float));

    /* Check allocations */
    if (!s->k_cache || !s->v_cache || !s->hidden || !s->normed || !s->buf_q ||
        !s->buf_k || !s->buf_v || !s->scores || !s->ffbuf || !s->logits)
        goto fail;

    /* SSM state caches (only for SSM models) */
    s->ssm_conv_state   = NULL;
    s->ssm_hidden_state = NULL;
    if (cfg->ssm_d_conv > 0 && cfg->ssm_d_inner > 0 && cfg->ssm_d_state > 0) {
        int d_conv  = cfg->ssm_d_conv;
        int d_inner = cfg->ssm_d_inner;
        int d_state = cfg->ssm_d_state;
        size_t conv_sz = (size_t)cfg->n_layer * d_inner * (d_conv > 0 ? d_conv - 1 : 0);
        size_t hid_sz  = (size_t)cfg->n_layer * d_state * d_inner;

        if (conv_sz > 0) {
            s->ssm_conv_state = (float*)calloc(conv_sz, sizeof(float));
            if (!s->ssm_conv_state) goto fail;
        }
        if (hid_sz > 0) {
            s->ssm_hidden_state = (float*)calloc(hid_sz, sizeof(float));
            if (!s->ssm_hidden_state) goto fail;
        }
        fprintf(stderr, "infer: SSM caches allocated (conv=%zu els, state=%zu els)\n",
                conv_sz, hid_sz);
    }

    return s;

fail:
    ct_infer_free(s);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_infer_forward — Single token forward pass
 *
 * hidden_in:  [n_embd] — input (token embedding or previous output)
 * hidden_out: [n_embd] — output hidden state (before final norm)
 * pos: position in the sequence (for RoPE + KV cache)
 *
 * Returns 0 on success, -1 on failure.
 * ═══════════════════════════════════════════════════════════════ */

int ct_infer_forward(ct_infer_state* s, int pos,
                     const float* hidden_in, float* hidden_out) {
    ct_infer_config* cfg = &s->w.config;
    const int L = cfg->n_layer;
    const int E = cfg->n_embd;
    const int H = cfg->n_head;
    const int HK = cfg->n_head_kv;
    const int HD = cfg->head_dim;
    const int n_groups = H / HK;  /* GQA groups */
    float rcp_sqrt_hd = 1.0f / sqrtf((float)HD);

    /* ─── Per-layer transformer blocks ─── */
    float* h = s->hidden;
    memcpy(h, hidden_in, (size_t)E * sizeof(float));

    for (int layer = 0; layer < L; layer++) {
        ct_infer_layer* lw = &s->w.layers[layer];

        /* ── SSM / Attention sub-block ── */

        /* Vulkan flag for batching (used in attention + FFN sections) */
        int use_vk = 0;

        if (lw->is_ssm) {
            /* ── SSM (Mamba) forward pass ── */
            /* Get per-layer SSM state caches */
            int d_inner = cfg->ssm_d_inner;
            int d_conv  = cfg->ssm_d_conv;
            int d_state = cfg->ssm_d_state;
            int c_stride = d_conv > 1 ? d_conv - 1 : 1;

            float* conv_state = s->ssm_conv_state
                ? s->ssm_conv_state + (size_t)layer * d_inner * c_stride
                : NULL;
            float* hid_state  = s->ssm_hidden_state
                ? s->ssm_hidden_state + (size_t)layer * d_state * d_inner
                : NULL;

            /* RMS norm (same norm used by SSM block as input normalization) */
            rms_norm(s->normed, h, lw->attn_norm, E, cfg->norm_rms_eps);

            /* SSM forward: normed → ssm_block → residual */
            ct_forward_ssm(s->ffbuf, s->normed, lw, cfg,
                          conv_state, hid_state);
            for (int i = 0; i < E; i++)
                h[i] += s->ffbuf[i];

        } else {
            /* ── Attention sub-block ── */

            /* RMS norm (s->normed is dedicated — no aliasing with bufs) */
            rms_norm(s->normed, h, lw->attn_norm, E, cfg->norm_rms_eps);

            /* Batch: Q, K, V all from the same normed input (skip Vulkan for layers beyond gpu_layers) */
#ifdef CT_VULKAN
            use_vk = (g_vk && layer < s->gpu_layers);
            if (use_vk) ct_vulkan_batch_begin(g_vk);
#endif
            __asm__ volatile("" ::: "memory");
            matmul(s->buf_q, s->normed, lw->attn_q, lw->t_q, E, H * HD);
            __asm__ volatile("" ::: "memory");
            matmul(s->buf_k, s->normed, lw->attn_k, lw->t_k, E, HK * HD);
            __asm__ volatile("" ::: "memory");
            matmul(s->buf_v, s->normed, lw->attn_v, lw->t_v, E, HK * HD);
#ifdef CT_VULKAN
            if (use_vk) ct_vulkan_batch_end(g_vk);
#endif
            /* Add QKV biases if present (Qwen2 uses them, LLaMA doesn't) */
            if (lw->attn_q_bias)
                for (int j = 0; j < H * HD; j++) s->buf_q[j] += lw->attn_q_bias[j];
            if (lw->attn_k_bias)
                for (int j = 0; j < HK * HD; j++) s->buf_k[j] += lw->attn_k_bias[j];
            if (lw->attn_v_bias)
                for (int j = 0; j < HK * HD; j++) s->buf_v[j] += lw->attn_v_bias[j];
            /* Apply RoPE to Q (once per head) */
            for (int hh = 0; hh < H; hh++)
                rope(s->buf_q + hh * HD, HD, pos, cfg->rope_freq_base);
            /* Apply RoPE to K (once per KV head) */
            for (int hh = 0; hh < HK; hh++)
                rope(s->buf_k + hh * HD, HD, pos, cfg->rope_freq_base);

            /* ── Store K, V into KV cache ── */
            /* Layout: cache[layer][head][pos][dim] — contiguous within (head, pos) */
            size_t layer_stride = (size_t)HK * HD * s->max_ctx;
            float* layer_k = s->k_cache + layer * layer_stride;
            float* layer_v = s->v_cache + layer * layer_stride;

            for (int hh = 0; hh < HK; hh++) {
                float* k_dst = layer_k + (size_t)hh * HD * s->max_ctx + (size_t)pos * HD;
                float* v_dst = layer_v + (size_t)hh * HD * s->max_ctx + (size_t)pos * HD;
                memcpy(k_dst, s->buf_k + hh * HD, (size_t)HD * sizeof(float));
                memcpy(v_dst, s->buf_v + hh * HD, (size_t)HD * sizeof(float));
            }

            /* ── Attention ── */
            /* attn_out overwrites buf_q (Q consumed per-head, then overwritten) */
            float* attn_out = s->buf_q;
            for (int hh = 0; hh < H; hh++) {
                int kg = hh / n_groups;        /* shared KV head index */
                float* Qh = s->buf_q + hh * HD;
                float* Kh = layer_k + (size_t)kg * HD * s->max_ctx;
                float* Vh = layer_v + (size_t)kg * HD * s->max_ctx;
                float* out_h = attn_out + hh * HD;

                /* Scores: score[p] = Qh·Kh[p] / sqrt(HD) */
                float max_score = -1e38f;
                for (int p = 0; p <= pos; p++) {
                    float* Kp = Kh + (size_t)p * HD;
                    float dot = 0.0f;
                    for (int d = 0; d < HD; d++)
                        dot += Qh[d] * Kp[d];
                    s->scores[p] = dot * rcp_sqrt_hd;
                    if (s->scores[p] > max_score) max_score = s->scores[p];
                }

                /* Softmax */
                float sum_exp = 0.0f;
                for (int p = 0; p <= pos; p++) {
                    s->scores[p] = expf(s->scores[p] - max_score);
                    sum_exp += s->scores[p];
                }
                float rcp_sum = 1.0f / (sum_exp + 1e-10f);
                for (int p = 0; p <= pos; p++)
                    s->scores[p] *= rcp_sum;

                /* Weighted sum: out_h += softmax[p] * Vh[p] */
                memset(out_h, 0, (size_t)HD * sizeof(float));
                for (int p = 0; p <= pos; p++) {
                    float sp = s->scores[p];
                    float* Vp = Vh + (size_t)p * HD;
                    for (int d = 0; d < HD; d++)
                        out_h[d] += sp * Vp[d];
                }
            }

            /* Attention output projection: attn_residual[E] = attn_out[H*HD] @ Wo[H*HD, E] */
            /* Reuse ffbuf for attn_residual (it's [n_ff] >= [n_embd]) */
#ifdef CT_VULKAN
            if (g_vk) ct_vulkan_batch_begin(g_vk);
#endif
            matmul(s->ffbuf, attn_out, lw->attn_out, lw->t_o, H * HD, E);
#ifdef CT_VULKAN
            if (g_vk) ct_vulkan_batch_end(g_vk);
#endif
            for (int i = 0; i < E; i++)
                h[i] += s->ffbuf[i];
        }

        /* ── FFN sub-block ── */

        /* RMS norm (dedicated buffer) */
        rms_norm(s->normed, h, lw->ffn_norm, E, cfg->norm_rms_eps);

        if (cfg->n_expert > 0) {
            /* ── MoE: router + expert dispatch ── */

            /* Router: ffbuf[n_expert] = normed @ ffn_gate (router weights) */
            matmul(s->ffbuf, s->normed, lw->ffn_gate, lw->t_g, E, cfg->n_expert);

            /* Softmax over router logits */
            float max_r = s->ffbuf[0];
            for (int i = 1; i < cfg->n_expert; i++)
                if (s->ffbuf[i] > max_r) max_r = s->ffbuf[i];
            float sum_exp = 0.0f;
            for (int i = 0; i < cfg->n_expert; i++) {
                s->ffbuf[i] = expf(s->ffbuf[i] - max_r);
                sum_exp += s->ffbuf[i];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-10f);
            for (int i = 0; i < cfg->n_expert; i++)
                s->ffbuf[i] *= inv_sum;

            /* Top-k expert selection (argmax k times, mark used with -1e10f) */
            #define CT_MOE_MAX_ROUTED 8
            int e_idx[CT_MOE_MAX_ROUTED];
            float e_w[CT_MOE_MAX_ROUTED];
            int n_routed = cfg->n_expert_per_token;
            if (n_routed > CT_MOE_MAX_ROUTED) n_routed = CT_MOE_MAX_ROUTED;
            if (n_routed > cfg->n_expert) n_routed = cfg->n_expert;
            if (n_routed < 1) n_routed = 1;
            for (int r = 0; r < n_routed; r++) {
                int best = 0;
                for (int i = 1; i < cfg->n_expert; i++)
                    if (s->ffbuf[i] > s->ffbuf[best]) best = i;
                e_idx[r] = best;
                e_w[r] = s->ffbuf[best];
                s->ffbuf[best] = -1e10f;
            }

            /* Accumulate weighted expert output into h */
            for (int r = 0; r < n_routed; r++) {
                int eid = e_idx[r];

                /* Expert gate: ffbuf = silu(normed @ expert_gate[eid]) */
                matmul(s->ffbuf, s->normed, lw->expert_gate[eid],
                       lw->t_eg[eid], E, cfg->n_ff);
                for (int i = 0; i < cfg->n_ff; i++)
                    s->ffbuf[i] = silu(s->ffbuf[i]);

                /* Expert up: buf_k = normed @ expert_up[eid] */
                matmul(s->buf_k, s->normed, lw->expert_up[eid],
                       lw->t_eu[eid], E, cfg->n_ff);

                /* Gate * up -> ffbuf */
                for (int i = 0; i < cfg->n_ff; i++)
                    s->ffbuf[i] *= s->buf_k[i];

                /* Expert down: buf_v = (gate*up) @ expert_down[eid] */
                matmul(s->buf_v, s->ffbuf, lw->expert_down[eid],
                       lw->t_ed[eid], cfg->n_ff, E);

                /* Weighted accumulate to residual */
                for (int i = 0; i < E; i++)
                    h[i] += e_w[r] * s->buf_v[i];
            }

        } else {
            /* ── Dense FFN (original) ── */

            /* Batch: gate + up (both read normed, write different buffers) */
#ifdef CT_VULKAN
            if (use_vk) ct_vulkan_batch_begin(g_vk);
#endif
            matmul(s->ffbuf, s->normed, lw->ffn_gate, lw->t_g, E, cfg->n_ff);
            matmul(s->buf_k, s->normed, lw->ffn_up, lw->t_u, E, cfg->n_ff);
#ifdef CT_VULKAN
            if (use_vk) ct_vulkan_batch_end(g_vk);
#endif

            /* SiLU gate output in-place */
            for (int i = 0; i < cfg->n_ff; i++)
                s->ffbuf[i] = silu(s->ffbuf[i]);

            /* Element-wise: gate*up in-place in ffbuf */
            for (int i = 0; i < cfg->n_ff; i++)
                s->ffbuf[i] *= s->buf_k[i];

            /* Down: buf_v[E] = (gate*up) @ Wdown */
#ifdef CT_VULKAN
            if (use_vk) ct_vulkan_batch_begin(g_vk);
#endif
            matmul(s->buf_v, s->ffbuf, lw->ffn_down, lw->t_d, cfg->n_ff, E);
#ifdef CT_VULKAN
            if (use_vk) ct_vulkan_batch_end(g_vk);
#endif
            for (int i = 0; i < E; i++)
                h[i] += s->buf_v[i];
        }
    }

    /* ── Copy result ── */
    memcpy(hidden_out, h, (size_t)E * sizeof(float));
    s->n_ctx = pos + 1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Sampling
 * ═══════════════════════════════════════════════════════════════ */

int ct_infer_sample(const float* logits, int n_vocab, float temp, int top_k, float top_p) {
    if (n_vocab <= 0) return 0;

    /* Greedy: pure argmax */
    if (temp < 0.001f) {
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }

    /* Determine pool size: use top_k if > 0, otherwise full vocab */
    int k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    if (k > 512) k = 512; /* cap for stack allocation */

    typedef struct { int idx; float val; } scored;
    scored top[512];
    int filled = 0;
    float inv_temp = 1.0f / temp;

    /* Collect top-k logits */
    for (int i = 0; i < n_vocab; i++) {
        float scaled = logits[i] * inv_temp;
        if (filled < k) {
            top[filled].idx = i;
            top[filled].val = scaled;
            filled++;
            if (filled == k) {
                /* Bubble smallest to top[0] */
                for (int a = 0; a < k; a++)
                    for (int b = a + 1; b < k; b++)
                        if (top[b].val < top[a].val) {
                            scored t = top[a]; top[a] = top[b]; top[b] = t;
                        }
            }
        } else if (scaled > top[0].val) {
            top[0].idx = i;
            top[0].val = scaled;
            for (int j = 1; j < k; j++)
                if (top[j].val < top[0].val) {
                    scored t = top[0]; top[0] = top[j]; top[j] = t;
                }
        }
    }

    /* Sort descending by logit value for top-p */
    for (int a = 0; a < filled; a++)
        for (int b = a + 1; b < filled; b++)
            if (top[b].val > top[a].val) {
                scored t = top[a]; top[a] = top[b]; top[b] = t;
            }

    /* Softmax (numerically stable) */
    float max_val = top[0].val;
    float sum = 0;
    for (int i = 0; i < filled; i++) {
        top[i].val = expf(top[i].val - max_val);
        sum += top[i].val;
    }

    /* Top-p (nucleus) filtering: keep smallest set with cumsum >= top_p */
    int cutoff = filled;
    if (top_p < 1.0f && top_p > 0.0f) {
        float cum = 0;
        for (int i = 0; i < filled; i++) {
            cum += top[i].val / sum;
            if (cum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    /* Renormalize over the filtered set */
    float sub_sum = 0;
    for (int i = 0; i < cutoff; i++) sub_sum += top[i].val;

    /* Sample from filtered distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    for (int i = 0; i < cutoff; i++) {
        cum += top[i].val / sub_sum;
        if (r < cum) return top[i].idx;
    }
    return top[cutoff - 1].idx;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_infer_generate — Autoregressive generation
 *
 * tokens[0..n_prompt-1]: input token IDs (prompt)
 * max_gen: max new tokens to generate
 * output_tokens: receives generated token IDs (n_prompt + new tokens)
 * Returns total tokens generated (n_prompt + new), or <0 on error.
 * ═══════════════════════════════════════════════════════════════ */

int ct_infer_generate(ct_infer_state* s,
                      const int* tokens, int n_prompt,
                      int max_gen, float temp, int eos_id,
                      int* output_tokens,
                      void (*on_token)(int token, void* ctx),
                      void* stream_ctx,
                      float top_p, float repeat_penalty, int top_k) {
    if (!s || !tokens || n_prompt < 1 || !output_tokens) return -1;

    /* Seed RNG once per generation */
    srand((unsigned int)time(NULL));

    ct_infer_config* cfg = &s->w.config;
    const int E = cfg->n_embd;

    /* Copy prompt tokens to output */
    for (int i = 0; i < n_prompt; i++)
        output_tokens[i] = tokens[i];

    /* Prefill: embed ALL prompt tokens and run forward pass for each.
     * Each position's hidden state starts fresh with its token embedding
     * (independent per-position residual streams). The KV cache accumulates
     * across positions so attention can look back at earlier tokens. */
    float* layer_out = s->buf_q; /* reuse — working buffer */
    for (int i = 0; i < n_prompt; i++) {
        embed_row(s->hidden, s->w.token_embd, s->w.t_embd, tokens[i], E);
        if (ct_infer_forward(s, i, s->hidden, layer_out) != 0)
            return -1;
    }

    int total = n_prompt;
    int rep_ctx = 64; /* number of recent tokens to apply repeat penalty to */
    if (rep_ctx > n_prompt) rep_ctx = n_prompt;

    /* Generation loop */
    for (int gen = 0; gen < max_gen; gen++) {
        /* Final RMS norm (into normed buffer to avoid aliasing matmul) */
        rms_norm(s->normed, layer_out, s->w.final_norm, E, cfg->norm_rms_eps);

        /* Output projection: logits = normed @ output_weight */
        if (s->w.output_weight) {
            matmul(s->logits, s->normed, s->w.output_weight,
                   s->w.t_out, E, cfg->n_vocab);
        } else {
            /* Weight tying: use token_embd */
            matmul(s->logits, s->normed, s->w.token_embd,
                   s->w.t_embd, E, cfg->n_vocab);
        }

        /* Apply repeat penalty */
        if (repeat_penalty > 1.001f) {
            int start = total - rep_ctx;
            if (start < 0) start = 0;
            for (int i = start; i < total; i++) {
                int tid = output_tokens[i];
                if (tid > 0 && tid < cfg->n_vocab) {
                    if (s->logits[tid] > 0)
                        s->logits[tid] /= repeat_penalty;
                    else
                        s->logits[tid] *= repeat_penalty;
                }
            }
        }

        /* Sample next token */
        int next = ct_infer_sample(s->logits, cfg->n_vocab, temp, top_k, top_p);

        if (next <= 0 || next >= cfg->n_vocab) next = 1; /* BOS fallback */

        /* Streaming callback: notify listener of each generated token */
        if (on_token) on_token(next, stream_ctx);

        /* Stop generation at EOS token (do NOT include EOS in output) */
        if (eos_id > 0 && next == eos_id) break;

        output_tokens[total++] = next;

        /* Check context limit */
        if (total >= s->max_ctx) break;

        /* Embed next token for next iteration */
        embed_row(s->hidden, s->w.token_embd, s->w.t_embd, next, E);

        /* Forward pass */
        if (ct_infer_forward(s, total - 1, s->hidden, layer_out) != 0)
            break;
    }

    return total;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_infer_free
 * ═══════════════════════════════════════════════════════════════ */

void ct_infer_free(ct_infer_state* s) {
    if (!s) return;
#ifdef CT_VULKAN
    if (s->vk_backend) {
        ct_vulkan_destroy((ct_vulkan_backend*)s->vk_backend);
        s->vk_backend = NULL;
    }
#endif
    /* Free expert weight arrays per layer */
    if (s->w.layers && s->w.config.n_expert > 0) {
        for (int i = 0; i < s->w.config.n_layer; i++) {
            ct_infer_layer* l = &s->w.layers[i];
            free(l->expert_gate);
            free(l->expert_up);
            free(l->expert_down);
            free(l->t_eg);
            free(l->t_eu);
            free(l->t_ed);
        }
    }
    free(s->w.layers);
    free(s->k_cache);
    free(s->v_cache);
    free(s->ssm_conv_state);
    free(s->ssm_hidden_state);
    free(s->hidden);
    free(s->normed);
    free(s->buf_q);
    free(s->buf_k);
    free(s->buf_v);
    free(s->scores);
    free(s->ffbuf);
    free(s->logits);
    free(s);
}

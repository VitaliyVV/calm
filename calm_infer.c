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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        l->attn_norm = (float*)find_tensor(gguf, name, &tmp_type);
        if (i == 0) fprintf(stderr, "infer: attn_norm type=%d (layer 0)\n", tmp_type);
        if (!l->attn_norm) {
            fprintf(stderr, "infer: missing blk.%d.attn_norm.weight\n", i);
            return -1;
        }

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        l->attn_q = (void*)find_tensor(gguf, name, &l->t_q);
        if (!l->attn_q) {
            fprintf(stderr, "infer: missing blk.%d.attn_q.weight\n", i);
            return -1;
        }
        if (i == 0) fprintf(stderr, "infer: blk.0.attn_q.weight type=%d\n", l->t_q);
        /* QKV bias — optional (Qwen2 uses them, LLaMA doesn't) */
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

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        l->ffn_up = (void*)find_tensor(gguf, name, &l->t_u);
        if (!l->ffn_up) {
            fprintf(stderr, "infer: missing blk.%d.ffn_up.weight\n", i);
            return -1;
        }
        if (i == 0) fprintf(stderr, "infer: blk.0.ffn_up.weight type=%d\n", l->t_u);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        l->ffn_down = (void*)find_tensor(gguf, name, &l->t_d);
        if (!l->ffn_down) {
            fprintf(stderr, "infer: missing blk.%d.ffn_down.weight\n", i);
            return -1;
        }
        if (i == 0) fprintf(stderr, "infer: blk.0.ffn_down.weight type=%d\n", l->t_d);
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_infer_create
 * ═══════════════════════════════════════════════════════════════ */

ct_infer_state* ct_infer_create(ct_gguf_context* gguf) {
    if (!gguf) return NULL;

    /* Initialize FP16 lookup table for quantized operations */
    ct_quant_init();

    ct_infer_state* s = (ct_infer_state*)calloc(1, sizeof(ct_infer_state));
    if (!s) return NULL;
    s->gguf = gguf;

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

    /* KV cache: cap context to reasonable size */
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

    if (pos < 2) {
        float rms = 0;
        for (int i = 0; i < E; i++) rms += h[i] * h[i];
        rms = sqrtf(rms / (float)E);
        fprintf(stderr, "[DBG] pos=%d PRE-LAYER h_rms=%.4f h[0]=%.4f h[1]=%.4f\n", pos, rms, h[0], h[1]);
    }

    for (int layer = 0; layer < L; layer++) {
        ct_infer_layer* lw = &s->w.layers[layer];

        /* ── Attention sub-block ── */

        /* RMS norm (s->normed is dedicated — no aliasing with bufs) */
        rms_norm(s->normed, h, lw->attn_norm, E, cfg->norm_rms_eps);

        /* Q, K, V all from the same normed input */
        matmul(s->buf_q, s->normed, lw->attn_q, lw->t_q, E, H * HD);
        matmul(s->buf_k, s->normed, lw->attn_k, lw->t_k, E, HK * HD);
        matmul(s->buf_v, s->normed, lw->attn_v, lw->t_v, E, HK * HD);
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

            /* Debug: print attention scores for head 0 at L0, pos 0..3 */
            if (layer == 0 && hh == 0 && pos <= 3) {
                fprintf(stderr, "[DBG] L0 h=0 pos=%d scores:", pos);
                for (int p = 0; p <= pos; p++)
                    fprintf(stderr, " %.4f", s->scores[p]);
                fprintf(stderr, "\n");
            }

            /* Weighted sum: out_h += softmax[p] * Vh[p] */
            memset(out_h, 0, (size_t)HD * sizeof(float));
            for (int p = 0; p <= pos; p++) {
                float sp = s->scores[p];
                float* Vp = Vh + (size_t)p * HD;
                for (int d = 0; d < HD; d++)
                    out_h[d] += sp * Vp[d];
            }
        }

        /* Debug: verify attention and track hidden state per layer */
        if (pos < 2) {
            float rms = 0;
            for (int i = 0; i < E; i++) rms += h[i] * h[i];
            rms = sqrtf(rms / (float)E);
            if (layer == 0 || layer == L-1 || layer == 1)
                fprintf(stderr, "[DBG] pos=%d L=%d h_rms=%.4f h[0]=%.4f h[1]=%.4f\n", pos, layer, rms, h[0], h[1]);
        }

        /* Debug: verify attention for layer 0, pos==0 (single-token: output should == V) */
        if (layer == 0 && pos == 0) {
            /* For head 0, kg=0: attn_out[0..63] should equal Vh[0..63] */
            float diff = 0;
            for (int d = 0; d < HD; d++) {
                /* Vh for kg=0, pos=0 */
                float v_val = layer_v[0 * HD * s->max_ctx + 0 * HD + d];
                diff += fabsf(attn_out[d] - v_val);
            }
            fprintf(stderr, "[DBG] layer0 pos0: attn_out[0] vs V[0] diff=%.6f (should be 0)\n", diff);
            fprintf(stderr, "[DBG]   attn_out[0..3]=%.4f %.4f %.4f %.4f  V[0..3]=%.4f %.4f %.4f %.4f\n",
                    attn_out[0], attn_out[1], attn_out[2], attn_out[3],
                    layer_v[0], layer_v[1], layer_v[2], layer_v[3]);
        }


        /* Attention output projection: attn_residual[E] = attn_out[H*HD] @ Wo[H*HD, E] */
        /* Reuse ffbuf for attn_residual (it's [n_ff] >= [n_embd]) */
        matmul(s->ffbuf, attn_out, lw->attn_out, lw->t_o, H * HD, E);
        for (int i = 0; i < E; i++)
            h[i] += s->ffbuf[i];

        /* ── FFN sub-block ── */

        /* RMS norm (dedicated buffer) */
        rms_norm(s->normed, h, lw->ffn_norm, E, cfg->norm_rms_eps);

        /* Gate: ffbuf[n_ff] = silu(normed @ Wgate) */
        matmul(s->ffbuf, s->normed, lw->ffn_gate, lw->t_g, E, cfg->n_ff);
        for (int i = 0; i < cfg->n_ff; i++)
            s->ffbuf[i] = silu(s->ffbuf[i]);

        /* Up: buf_k[n_ff] = normed @ Wup */
        matmul(s->buf_k, s->normed, lw->ffn_up, lw->t_u, E, cfg->n_ff);

        /* Element-wise: buf_k[i] *= gate[i] (gate*up in-place) */
        for (int i = 0; i < cfg->n_ff; i++)
            s->ffbuf[i] *= s->buf_k[i];

        /* Down: buf_v[E] = (gate*up) @ Wdown */
        matmul(s->buf_v, s->ffbuf, lw->ffn_down, lw->t_d, cfg->n_ff, E);
        for (int i = 0; i < E; i++)
            h[i] += s->buf_v[i];

    }

    /* ── Copy result ── */
    memcpy(hidden_out, h, (size_t)E * sizeof(float));
    s->n_ctx = pos + 1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Sampling
 * ═══════════════════════════════════════════════════════════════ */

int ct_infer_sample(const float* logits, int n_vocab, float temp) {
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

    /* Top-k sampling: softmax over the k highest logits, sample from dist */
    #define CT_SAMPLE_TOP_K 40

    typedef struct { int idx; float val; } scored;
    scored top[CT_SAMPLE_TOP_K];
    int filled = 0;
    float inv_temp = 1.0f / temp;

    /* Collect top-k logits (option 0 baseline to init) */
    for (int i = 0; i < n_vocab; i++) {
        float scaled = logits[i] * inv_temp;
        if (filled < CT_SAMPLE_TOP_K) {
            top[filled].idx = i;
            top[filled].val = scaled;
            filled++;
            if (filled == CT_SAMPLE_TOP_K) {
                /* Bubble smallest to top[0] */
                for (int a = 0; a < CT_SAMPLE_TOP_K; a++)
                    for (int b = a + 1; b < CT_SAMPLE_TOP_K; b++)
                        if (top[b].val < top[a].val) {
                            scored t = top[a]; top[a] = top[b]; top[b] = t;
                        }
            }
        } else if (scaled > top[0].val) {
            /* Replace smallest in heap */
            top[0].idx = i;
            top[0].val = scaled;
            /* Sink smallest to correct position */
            for (int j = 1; j < CT_SAMPLE_TOP_K; j++)
                if (top[j].val < top[0].val) {
                    scored t = top[0]; top[0] = top[j]; top[j] = t;
                }
        }
    }

    /* Softmax */
    float max_val = top[0].val;
    for (int i = 1; i < filled; i++)
        if (top[i].val > max_val) max_val = top[i].val;

    float sum_exp = 0.0f;
    for (int i = 0; i < filled; i++) {
        top[i].val = expf(top[i].val - max_val);
        sum_exp += top[i].val;
    }

    /* Sample from categorical distribution */
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < filled; i++) {
        cum += top[i].val / sum_exp;
        if (r < cum) return top[i].idx;
    }
    return top[filled - 1].idx;
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
                      int* output_tokens) {
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

    /* Generation loop */
    for (int gen = 0; gen < max_gen; gen++) {
        /* Final RMS norm (into normed buffer to avoid aliasing matmul) */
        rms_norm(s->normed, layer_out, s->w.final_norm, E, cfg->norm_rms_eps);

        /* Debug: check hidden state norm and first 5 values before output proj */
        if (gen == 0) {
            float h_norm = 0;
            for (int i = 0; i < E; i++) h_norm += layer_out[i] * layer_out[i];
            h_norm = sqrtf(h_norm / (float)E);
            float n_norm = 0;
            for (int i = 0; i < E; i++) n_norm += s->normed[i] * s->normed[i];
            n_norm = sqrtf(n_norm / (float)E);
            fprintf(stderr, "[DBG] layer_out rms_norm=%.4f, normed rms=%.4f\n", h_norm, n_norm);
            fprintf(stderr, "[DBG] layer_out[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
                    layer_out[0], layer_out[1], layer_out[2], layer_out[3], layer_out[4]);
            fprintf(stderr, "[DBG] normed[0..4]:   %.4f %.4f %.4f %.4f %.4f\n",
                    s->normed[0], s->normed[1], s->normed[2], s->normed[3], s->normed[4]);
        }

        /* Output projection: logits = normed @ output_weight */
        if (s->w.output_weight) {
            matmul(s->logits, s->normed, s->w.output_weight,
                   s->w.t_out, E, cfg->n_vocab);
        } else {
            /* Weight tying: use token_embd */
            matmul(s->logits, s->normed, s->w.token_embd,
                   s->w.t_embd, E, cfg->n_vocab);
        }
        /* Sample next token */
        int next = ct_infer_sample(s->logits, cfg->n_vocab, temp);

        /* Debug: logit stats + top-5 */
        if (gen < 5) {
            float min_l = 1e38f, max_l = -1e38f, sum_l = 0;
            int top5[5] = {0};
            float top5v[5] = {-1e38f};
            for (int i = 0; i < cfg->n_vocab; i++) {
                if (s->logits[i] < min_l) min_l = s->logits[i];
                if (s->logits[i] > max_l) max_l = s->logits[i];
                sum_l += s->logits[i];
                for (int t = 0; t < 5; t++) {
                    if (s->logits[i] > top5v[t]) {
                        for (int t2 = 4; t2 > t; t2--) {
                            top5[t2] = top5[t2-1];
                            top5v[t2] = top5v[t2-1];
                        }
                        top5[t] = i;
                        top5v[t] = s->logits[i];
                        break;
                    }
                }
            }
            fprintf(stderr, "\n[DBG] gen=%d token=%d min=%.2f max=%.2f mean=%.4f\n",
                    gen, next, min_l, max_l, sum_l / cfg->n_vocab);
            fprintf(stderr, "[DBG] top5: %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f) %d(%.2f)\n",
                    top5[0], top5v[0], top5[1], top5v[1], top5[2], top5v[2],
                    top5[3], top5v[3], top5[4], top5v[4]);
        }

        if (next <= 0 || next >= cfg->n_vocab) next = 1; /* BOS fallback */

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
    free(s->w.layers);
    free(s->k_cache);
    free(s->v_cache);
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

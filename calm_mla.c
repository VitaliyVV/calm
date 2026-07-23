/**
 * calm_mla.c — Multi-head Latent Attention (DeepSeek2) Forward Pass
 *
 * MLA compresses K/V into a low-rank latent space:
 *
 *   c_kv = x @ Wkv_a          → compressed latent + RoPE part
 *   k_rope = rope(c_kv_rope)  → decoupled RoPE
 *   cache stores: [c | k_rope] per position = [dc + dr]
 *
 * Two weight formats supported:
 *   Decomposed: Q = (x @ Wq_a) @ Wq_b,  KV = x @ Wkv_a
 *   Fused:      Q =  x @ Wq,             KV = x @ Wkv_a_mqa (+ latent norm)
 *
 * Scoring uses absorption trick (F32) or per-position matmul (quantized):
 *
 *   F32: absorbed_q_h = Q_nope_h @ Wk_b_h  →  score = dot(absorbed_q_h, c[p])
 *         attn_out_h = c_weighted @ Wv_b_h  →  one matmul
 *
 *   Non-F32: K_nope = c[p] @ Wk_b_h per position (via matmul dispatch)
 *            V = c[p] @ Wv_b_h per position
 *
 * Wkv_b is stored column-major [dc, H*(dn+dv)]: column c starts at byte c * col_size.
 */

#include "calm_mla.h"
#include "calm_gguf.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* Quant block constants (from calm_infer.c) */
#define CT_QK_K 256

/* ── Forward declarations from calm_infer.c ── */
extern void matmul(float* y, const float* x, const void* w, int type, int I, int O);
extern void rope(float* buf, int n, int pos, float base);
extern void rms_norm(float* y, const float* x, const float* w, int n, float eps);

/* ── F32 fast path: absorption trick ── */
static void mla_forward_f32(float* buf_q, const float* normed,
                            const ct_infer_layer* lw,
                            const ct_infer_config* cfg,
                            float* k_cache, int pos, int max_ctx) {
    const int E = cfg->n_embd;
    const int H = cfg->n_head;
    const int HK = cfg->n_head_kv;
    const int dc = cfg->mla_kv_lora_rank;
    const int dr = cfg->mla_qk_rope_head_dim;
    const int dn = cfg->mla_qk_nope_head_dim;
    const int dv = cfg->mla_v_head_dim;
    const int q_rank = cfg->mla_q_lora_rank > 0 ? cfg->mla_q_lora_rank : dc;
    const int hd = dn + dr;
    const int total_per_head = dn + dv;
    const int total_cols = H * total_per_head;
    float rcp_sqrt_hd = 1.0f / sqrtf((float)hd);
    int n_groups = H / HK;
    if (n_groups < 1) n_groups = 1;

    float q_a_buf[2048];  /* q_lora_rank max */
    float kv_a_buf[1024]; /* dc + dr max */
    float absorbed_q[512]; /* dc max */
    float c_weighted[512]; /* dc max */
    float kv_b_cur[4096];  /* H*(dn+dv) max */

    /* Step 1: Q projection — fused or decomposed */
    int fused_q = (lw->attn_q != NULL);
    if (fused_q) {
        /* Fused: Q = normed @ attn_q  [E, H*hd] */
        matmul(buf_q, normed, lw->attn_q, lw->t_q, E, H * hd);
    } else {
        /* Decomposed: Q = (normed @ Wq_a) @ Wq_b */
        matmul(q_a_buf, normed, lw->attn_q_a, lw->t_qa, E, q_rank);
        if (lw->attn_q_norm) {
            float* qn = q_a_buf;
            float sum_sq = 0.0f;
            for (int i = 0; i < q_rank; i++) sum_sq += qn[i] * qn[i];
            float scale = 1.0f / sqrtf(sum_sq / (float)q_rank + 1e-6f);
            for (int i = 0; i < q_rank; i++) qn[i] *= scale;
        }
        matmul(buf_q, q_a_buf, lw->attn_q_b, lw->t_qb, q_rank, H * hd);
    }

    /* Q-nope norm (for decomposed Q nope part) */
    if (lw->attn_q_norm && !fused_q) {
        for (int hh = 0; hh < H; hh++) {
            float* qh = buf_q + hh * hd;
            float sum_sq = 0.0f;
            for (int d = 0; d < dn; d++) sum_sq += qh[d] * qh[d];
            float scale = 1.0f / sqrtf(sum_sq / (float)dn + 1e-6f);
            for (int d = 0; d < dn; d++) qh[d] *= scale;
        }
    }

    /* RoPE on Q rope part */
    for (int hh = 0; hh < H; hh++)
        rope(buf_q + hh * hd + dn, dr, pos, cfg->rope_freq_base);

    /* Step 2: KV latent — fused mqa or decomposed */
    if (lw->attn_kv_a_mqa) {
        /* Fused: c_kv = normed @ attn_kv_a_mqa  [E, dc+dr] */
        matmul(kv_a_buf, normed, lw->attn_kv_a_mqa, lw->t_kva_mqa, E, dc + dr);
    } else {
        matmul(kv_a_buf, normed, lw->attn_kv_a, lw->t_kva, E, dc + dr);
    }

    /* KV latent norm (applied to full latent before splitting) */
    if (lw->attn_kv_a_norm) {
        rms_norm(kv_a_buf, kv_a_buf, lw->attn_kv_a_norm, dc + dr, cfg->norm_rms_eps);
    }

    const float* c = kv_a_buf;
    const float* k_rope_raw = c + dc;

    float k_rope_cur[64];
    memcpy(k_rope_cur, k_rope_raw, (size_t)dr * sizeof(float));
    rope(k_rope_cur, dr, pos, cfg->rope_freq_base);

    for (int d = 0; d < dc; d++)
        k_cache[(size_t)d * max_ctx + pos] = c[d];
    for (int d = 0; d < dr; d++)
        k_cache[(size_t)(dc + d) * max_ctx + pos] = k_rope_cur[d];

    /* Step 3: K_nope norm (per-head, applied to Wkv_b output) */
    if (lw->attn_k_norm) {
        matmul(kv_b_cur, c, lw->attn_kv_b, lw->t_kvb, dc, total_cols);
        for (int hh = 0; hh < HK; hh++) {
            float* kh = kv_b_cur + hh * total_per_head;
            float sum_sq = 0.0f;
            for (int d = 0; d < dn; d++) sum_sq += kh[d] * kh[d];
            float scale = 1.0f / sqrtf(sum_sq / (float)dn + 1e-6f);
            for (int d = 0; d < dn; d++) kh[d] *= scale;
        }
    }

    /* Wkv_b is [dc, total_cols] column-major:
     *   element (row=k, col=c) = w_f32[c * dc + k] */
    const float* w_kv_b = (const float*)lw->attn_kv_b;

    /* Step 4: Per-head attention via absorption */
    for (int hh = 0; hh < H; hh++) {
        int kg = hh / n_groups;
        float* Qh = buf_q + hh * hd;
        size_t head_col = (size_t)kg * total_per_head;

        /* 4a. Absorb: absorbed_q[k] = Σ_d Q_nope[d] * Wkv_b[k][head_col+d]
         *     Column-major: Wkv_b[k][c] = w_f32[c * dc + k] */
        memset(absorbed_q, 0, (size_t)dc * sizeof(float));
        for (int d = 0; d < dn; d++) {
            float qd = Qh[d];
            const float* col = w_kv_b + (head_col + d) * dc;
            for (int k = 0; k < dc; k++)
                absorbed_q[k] += qd * col[k];
        }

        /* 4b. Score all positions */
        float* scores = kv_b_cur;
        float max_score = -1e38f;
        const float* Q_rope = Qh + dn;

        for (int p = 0; p <= pos; p++) {
            float dot_nope = 0.0f;
            const float* c_p = k_cache + (size_t)p;
            for (int k = 0; k < dc; k++)
                dot_nope += absorbed_q[k] * c_p[(size_t)k * max_ctx];

            float dot_rope = 0.0f;
            const float* kr_p = k_cache + (size_t)dc * max_ctx + p;
            for (int d = 0; d < dr; d++)
                dot_rope += Q_rope[d] * kr_p[(size_t)d * max_ctx];

            scores[p] = (dot_nope + dot_rope) * rcp_sqrt_hd;
            if (scores[p] > max_score) max_score = scores[p];
        }

        /* 4c. Softmax */
        float sum_exp = 0.0f;
        for (int p = 0; p <= pos; p++) {
            scores[p] = expf(scores[p] - max_score);
            sum_exp += scores[p];
        }
        float rcp_sum = 1.0f / (sum_exp + 1e-10f);
        for (int p = 0; p <= pos; p++)
            scores[p] *= rcp_sum;

        /* 4d. Weighted latent: c_weighted[k] = Σ_p score[p] * c[p][k] */
        memset(c_weighted, 0, (size_t)dc * sizeof(float));
        for (int p = 0; p <= pos; p++) {
            float sp = scores[p];
            const float* c_p = k_cache + (size_t)p;
            for (int k = 0; k < dc; k++)
                c_weighted[k] += sp * c_p[(size_t)k * max_ctx];
        }

        /* 4e. V = c_weighted @ Wv_b_h
         *     Wv_b_h columns start at head_col + dn
         *     out_h[d] = Σ_k c_weighted[k] * w_f32[(v_col+d) * dc + k] */
        float* out_h = buf_q + hh * dv;
        size_t v_col = head_col + dn;
        for (int d = 0; d < dv; d++) {
            float sum = 0.0f;
            const float* vcol = w_kv_b + (v_col + d) * dc;
            for (int k = 0; k < dc; k++)
                sum += c_weighted[k] * vcol[k];
            out_h[d] = sum;
        }
    }
}

/* ── Forward declarations from calm_infer.c for per-block dequant ── */
extern void deq_q2_K(const void* b, float* out);
extern void deq_q3_K(const void* b, float* out);
extern void deq_q4_K(const void* b, float* out);
extern void deq_q5_0(const void* b, float* out);
extern void deq_q4_1(const void* b, float* out);
extern void deq_q6_K(const void* b, float* out);
extern void deq_q8_K(const void* b, float* out);
extern void deq_iq4_nl(const void* b, float* out);

/* ── General path: absorption-based with per-row dequant for quantized Wkv_b ── */
static void mla_forward_general(float* buf_q, const float* normed,
                                const ct_infer_layer* lw,
                                const ct_infer_config* cfg,
                                float* k_cache, int pos, int max_ctx) {
    const int E = cfg->n_embd;
    const int H = cfg->n_head;
    const int HK = cfg->n_head_kv;
    const int dc = cfg->mla_kv_lora_rank;
    const int dr = cfg->mla_qk_rope_head_dim;
    const int dn = cfg->mla_qk_nope_head_dim;
    const int dv = cfg->mla_v_head_dim;
    const int q_rank = cfg->mla_q_lora_rank > 0 ? cfg->mla_q_lora_rank : dc;
    const int hd = dn + dr;
    const int total_per_head = dn + dv;
    const int total_cols = H * total_per_head;
    const int blocks_per_row = total_cols / CT_QK_K;
    float rcp_sqrt_hd = 1.0f / sqrtf((float)hd);
    int n_groups = H / HK;
    if (n_groups < 1) n_groups = 1;

    float q_a_buf[2048];
    float kv_a_buf[1024];
    float kv_b_cur[4096]; /* scores up to max_ctx + K_norm output (max H*total_per_head=4096) */

    /* Per-block byte stride in Wkv_b for quantized types */
    int block_stride = ct_gguf_tensor_size(lw->t_kvb, 1, (uint64_t[]){ (uint64_t)CT_QK_K });

    /* Select dequant function pointer based on t_kvb */
    void (*deq_fn)(const void*, float*) = NULL;
    switch (lw->t_kvb) {
        case CT_GGUF_TYPE_Q2_K: deq_fn = deq_q2_K; break;
        case CT_GGUF_TYPE_Q3_K: deq_fn = deq_q3_K; break;
        case CT_GGUF_TYPE_Q4_K: deq_fn = deq_q4_K; break;
        case CT_GGUF_TYPE_Q5_0: deq_fn = deq_q5_0; break;
        case CT_GGUF_TYPE_Q4_1: deq_fn = deq_q4_1; break;
        case CT_GGUF_TYPE_Q6_K: deq_fn = deq_q6_K; break;
        case CT_GGUF_TYPE_Q8_K: deq_fn = deq_q8_K; break;
        default: fprintf(stderr, "[MLA] unsupported Wkv_b type %d\n", lw->t_kvb); break;
    }

    /* Step 1: Q — fused or decomposed */
    int fused_q = (lw->attn_q != NULL);
    if (fused_q) {
        matmul(buf_q, normed, lw->attn_q, lw->t_q, E, H * hd);
    } else {
        matmul(q_a_buf, normed, lw->attn_q_a, lw->t_qa, E, q_rank);
        if (lw->attn_q_norm) {
            float* qn = q_a_buf;
            float sum_sq = 0, scale;
            for (int i = 0; i < q_rank; i++) sum_sq += qn[i] * qn[i];
            scale = 1.0f / sqrtf(sum_sq / (float)q_rank + 1e-6f);
            for (int i = 0; i < q_rank; i++) qn[i] *= scale;
        }
        matmul(buf_q, q_a_buf, lw->attn_q_b, lw->t_qb, q_rank, H * hd);
    }

    /* Q-nope norm */
    if (lw->attn_q_norm && !fused_q) {
        for (int hh = 0; hh < H; hh++) {
            float* qh = buf_q + hh * hd;
            float sum_sq = 0.0f;
            for (int d = 0; d < dn; d++) sum_sq += qh[d] * qh[d];
            float scale = 1.0f / sqrtf(sum_sq / (float)dn + 1e-6f);
            for (int d = 0; d < dn; d++) qh[d] *= scale;
        }
    }

    /* RoPE on Q rope part */
    for (int hh = 0; hh < H; hh++)
        rope(buf_q + hh * hd + dn, dr, pos, cfg->rope_freq_base);

    /* Step 2: KV latent — fused mqa or decomposed */
    if (lw->attn_kv_a_mqa) {
        matmul(kv_a_buf, normed, lw->attn_kv_a_mqa, lw->t_kva_mqa, E, dc + dr);
    } else {
        matmul(kv_a_buf, normed, lw->attn_kv_a, lw->t_kva, E, dc + dr);
    }

    /* KV latent norm: weight tensor has dc elements (nope part only), not dc+dr */
    if (lw->attn_kv_a_norm) {
        rms_norm(kv_a_buf, kv_a_buf, lw->attn_kv_a_norm, dc, cfg->norm_rms_eps);
    }

    const float* c = kv_a_buf;
    const float* k_rope_raw = c + dc;

    float k_rope_cur[64];
    memcpy(k_rope_cur, k_rope_raw, (size_t)dr * sizeof(float));
    rope(k_rope_cur, dr, pos, cfg->rope_freq_base);

    for (int d = 0; d < dc; d++)
        k_cache[(size_t)d * max_ctx + pos] = c[d];
    for (int d = 0; d < dr; d++)
        k_cache[(size_t)(dc + d) * max_ctx + pos] = k_rope_cur[d];

    /* Step 3: K_nope norm (per-head) */
    if (lw->attn_k_norm) {
        /* This path uses the same absorption approach but for K_norm.
         * For now, dispatch through matmul which is correct for full row access
         * (not column-sliced). K_norm uses the entire Wkv_b [dc, H*total_per_head]
         * but produces per-head K_nope which is then normed. */
        matmul(kv_b_cur, c, lw->attn_kv_b, lw->t_kvb, dc, H * total_per_head);
        for (int hh = 0; hh < HK; hh++) {
            float* kh = kv_b_cur + hh * total_per_head;
            float sum_sq = 0.0f;
            for (int d = 0; d < dn; d++) sum_sq += kh[d] * kh[d];
            float scale = 1.0f / sqrtf(sum_sq / (float)dn + 1e-6f);
            for (int d = 0; d < dn; d++) kh[d] *= scale;
        }
    }

    /* Step 4: Per-head attention via absorption (dequantize Wkv_b per-row on the fly)
     *
     * Wkv_b is stored row-major as [dc, total_cols]. Each row k has blocks_per_row blocks.
     * Head group kg uses columns [kg*total_per_head .. (kg+1)*total_per_head-1].
     * Since total_per_head = 256 = CT_QK_K, this is exactly one block per row per head group.
     *
     * K portion of head group kg: block kg of each row, sub-indices 0..dn-1
     * V portion of head group kg: block kg of each row, sub-indices dn..dn+dv-1
     */
    for (int hh = 0; hh < H; hh++) {
        int kg = hh / n_groups;
        float* Qh = buf_q + hh * hd;
        const float* Q_rope = Qh + dn;

        /* Step 4a: Absorb Q_nope into K columns
         * absorbed_q[k] = Σ_{d=0}^{dn-1} Qh[d] * Wkv_b[k][kg*256 + d]
         * Each row k's element is at block k*blocks_per_row + kg, sub-index d */
        float absorbed_q[512]; /* dc max */
        memset(absorbed_q, 0, (size_t)dc * sizeof(float));
        if (deq_fn) {
            for (int k = 0; k < dc; k++) {
                const void* bp = (const uint8_t*)lw->attn_kv_b + (size_t)(k * blocks_per_row + kg) * block_stride;
                float buf[256];
                deq_fn(bp, buf);
                float sum = 0.0f;
                for (int d = 0; d < dn; d++)
                    sum += Qh[d] * buf[d];
                absorbed_q[k] = sum;
            }
        }

        /* Step 4b: Scores (dot with cache for all positions) */
        float* scores = kv_b_cur;
        float max_score = -1e38f;
        for (int p = 0; p <= pos; p++) {
            float dot_nope = 0.0f;
            for (int k = 0; k < dc; k++)
                dot_nope += absorbed_q[k] * k_cache[(size_t)k * max_ctx + p];

            float dot_rope = 0.0f;
            const float* kr_p = k_cache + (size_t)dc * max_ctx + p;
            for (int d = 0; d < dr; d++)
                dot_rope += Q_rope[d] * kr_p[(size_t)d * max_ctx];

            scores[p] = (dot_nope + dot_rope) * rcp_sqrt_hd;
            if (scores[p] > max_score) max_score = scores[p];
        }

        /* Step 4c: Softmax */
        float sum_exp = 0.0f;
        for (int p = 0; p <= pos; p++) {
            scores[p] = expf(scores[p] - max_score);
            sum_exp += scores[p];
        }
        float rcp_sum = 1.0f / (sum_exp + 1e-10f);
        for (int p = 0; p <= pos; p++)
            scores[p] *= rcp_sum;

        /* Step 4d: Weighted cache sum for V absorption
         * weighted_c[k] = Σ_p scores[p] * k_cache[k * max_ctx + p] */
        float weighted_c[512]; /* dc max */
        memset(weighted_c, 0, (size_t)dc * sizeof(float));
        for (int p = 0; p <= pos; p++) {
            float sp = scores[p];
            for (int k = 0; k < dc; k++)
                weighted_c[k] += sp * k_cache[(size_t)k * max_ctx + p];
        }

        /* Step 4e: V absorption
         * out_h[d] = Σ_k weighted_c[k] * Wkv_b[k][kg*256 + dn + d] */
        float* out_h = buf_q + hh * dv;
        memset(out_h, 0, (size_t)dv * sizeof(float));
        if (deq_fn) {
            for (int k = 0; k < dc; k++) {
                const void* bp = (const uint8_t*)lw->attn_kv_b + (size_t)(k * blocks_per_row + kg) * block_stride;
                float buf[256];
                deq_fn(bp, buf);
                float wk = weighted_c[k];
                for (int d = 0; d < dv; d++)
                    out_h[d] += wk * buf[dn + d];
            }
        }
    }
}

/* ── Public API: dispatch F32 fast path vs general quantized path ── */
void ct_forward_mla(float* buf_q, const float* normed,
                    const ct_infer_layer* lw,
                    const ct_infer_config* cfg,
                    float* k_cache, int pos, int max_ctx) {
    if (lw->t_kvb == CT_GGUF_TYPE_F32) {
        mla_forward_f32(buf_q, normed, lw, cfg, k_cache, pos, max_ctx);
    } else {
        mla_forward_general(buf_q, normed, lw, cfg, k_cache, pos, max_ctx);
    }
}

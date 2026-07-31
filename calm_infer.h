/**
 * calm_infer.h — Calm Native Inference Engine
 *
 * LLaMA-family transformer forward pass with quantized weights.
 * Supports F32, Q8_0, Q4_0, BQ1_0, TQ1_0 matmul dispatch.
 *
 * Usage:
 *   ct_gguf_context* gguf = ct_gguf_open("model.gguf");
 *   ct_infer_state* s = ct_infer_create(gguf, 0, 0);
 *   ct_infer_generate(s, tokens, n_prompt, 128, output, sizeof(output));
 *   ct_infer_free(s);
 *   ct_gguf_close(gguf);
 */
#ifndef CALM_INFER_H
#define CALM_INFER_H

#include "calm_gguf.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Model config (extracted from GGUF metadata) ─── */
typedef struct {
    int   n_layer;
    int   n_embd;
    int   n_head;
    int   n_head_kv;
    int   n_ff;
    int   n_ctx_max;
    int   n_vocab;
    int   head_dim;
    int   n_expert;           /* number of experts (0 = dense model) */
    int   n_expert_per_token; /* top-k experts per token (e.g. 2)   */
    float norm_rms_eps;
    float rope_freq_base;
    /* ── SSM (Mamba) config (0 = not an SSM model) ── */
    int   ssm_d_conv;         /* ssm.conv_kernel (e.g. 4)          */
    int   ssm_d_inner;        /* ssm.inner_size (typically 2*n_embd) */
    int   ssm_d_state;        /* ssm.state_size (e.g. 16)          */
    int   ssm_dt_rank;        /* ssm.time_step_rank (e.g. 256)     */
    int   ssm_dt_b_c_rms;     /* ssm.dt_b_c_rms (Jamba-style norm) */
    /* ── MLA (DeepSeek2) config (0 = standard MHA/GQA) ── */
    int   mla_kv_lora_rank;    /* attention.kv_lora_rank (e.g. 512)  */
    int   mla_q_lora_rank;     /* attention.q_lora_rank (=0 if tied) */
    int   mla_qk_nope_head_dim;/* attention.key_nope_head_dim         */
    int   mla_qk_rope_head_dim;/* attention.key_rope_head_dim (e.g.64)*/
    int   mla_v_head_dim;      /* attention.value_head_dim            */
    int   n_shared_expert;     /* expert_count_shared (e.g. 2)       */
    int   n_expert_ff;         /* expert_feed_forward_length (e.g.1408)*/
    int   n_shared_ff;         /* shared expert FF dim (e.g.2816=2ff) */
    int   leading_dense_blocks;/* leading_dense_block_count (e.g. 1) */
} ct_infer_config;

/* ─── Per-layer weight pointers into mmap ─── */
typedef struct {
    /* Layer type: 0 = attention, 1 = SSM (Mamba), 2 = MLA (DeepSeek2) */
    int is_ssm;
    int is_mla;

    /* ── Attention weights (used when is_ssm == 0) ── */
    float* attn_norm;          /* F32 — norm weights always float */
    void*  attn_q;    int t_q;
    float* attn_q_bias;        /* Q bias (F32), may be NULL */
    void*  attn_k;    int t_k;
    float* attn_k_bias;        /* K bias (F32), may be NULL */
    void*  attn_v;    int t_v;
    float* attn_v_bias;        /* V bias (F32), may be NULL */
    void*  attn_out;  int t_o;

    /* ── SSM weights (used when is_ssm == 1) ── */
    void*  ssm_in;      int t_ssm_in;       /* [n_embd, 2*d_inner]     */
    void*  ssm_conv1d;  int t_ssm_conv1d;   /* [d_conv, d_inner]       */
    float* ssm_conv1d_b;                    /* [d_inner] bias          */
    void*  ssm_x;       int t_ssm_x;        /* [d_inner, dt_rank+2*d_state] */
    float* ssm_dt_norm;                     /* [dt_rank] optional norm */
    float* ssm_b_norm;                      /* [d_state] optional norm */
    float* ssm_c_norm;                      /* [d_state] optional norm */
    void*  ssm_dt;      int t_ssm_dt;       /* [dt_rank, d_inner] + bias */
    float* ssm_dt_b;                        /* [d_inner] bias          */
    void*  ssm_a;       int t_ssm_a;        /* [d_state, d_inner]      */
    void*  ssm_d;       int t_ssm_d;        /* [d_inner]               */
    void*  ssm_out;     int t_ssm_out;      /* [d_inner, n_embd]       */

    /* ── MLA (DeepSeek2) weights (used when is_mla == 1) ── */
    void*  attn_q_a;    int t_qa;   /* [n_embd, q_lora_rank] decomposed   */
    void*  attn_q_b;    int t_qb;   /* [q_lora_rank, n_head * head_dim]   */
    float* attn_q_norm;             /* Q nope norm (optional)             */
    void*  attn_kv_a;   int t_kva;  /* [n_embd, kv_lora_rank + rope_dim]  */
    void*  attn_kv_b;   int t_kvb;  /* [kv_lora_rank, n_head * (nope+v)]  */
    float* attn_k_norm;             /* K nope norm (optional)             */
    /* Modern GGUF format: fused Q + MQA KV latent + KV norm */
    void*  attn_kv_a_mqa;  int t_kva_mqa; /* [n_embd, dc+dr] fused        */
    float* attn_kv_a_norm;               /* RMS norm for KV latent [dc+dr]*/
    /* ── Shared expert FFN (DeepSeekMoE, used when n_shared_expert > 0) ── */
    void*  shared_gate; int t_sg;
    void*  shared_up;   int t_su;
    void*  shared_down; int t_sd;

    /* ── FFN weights (shared by both layer types) ── */
    float* ffn_norm;
    void*  ffn_gate;  int t_g;  /* dense: SwiGLU gate; MoE: router weights */
    void*  ffn_up;    int t_u;
    void*  ffn_down;  int t_d;
    /* MoE expert FFN weights (NULL when model is dense).
     * Per-expert pointers (format A) or stacked 3D tensors (format B). */
    void** expert_gate;  int* t_eg;  /* [n_expert] (per-expert) */
    void** expert_up;    int* t_eu;  /* [n_expert] (per-expert) */
    void** expert_down;  int* t_ed;  /* [n_expert] (per-expert) */
    void*  expert_gate_exps;  int t_ege; /* [I, O, n_exp] stacked 3D   */
    void*  expert_up_exps;    int t_eue; /* [I, O, n_exp] stacked 3D   */
    void*  expert_down_exps;  int t_ede; /* [O, I, n_exp] stacked 3D   */
    float* expert_gate_inp;              /* router [I, n_exp] (optional)*/
} ct_infer_layer;

/* ─── Full model weights ─── */
typedef struct {
    ct_infer_config config;
    void*  token_embd;   int t_embd;
    void*  output_weight; int t_out;
    float* final_norm;
    ct_infer_layer* layers;
} ct_infer_weights;

/* ─── Inference state ─── */
typedef struct {
    ct_gguf_context* gguf;
    ct_infer_weights w;
    int n_ctx;       /* current sequence length */
    int max_ctx;     /* KV cache capacity      */
    float* k_cache;  /* [n_layer, n_head_kv, head_dim, max_ctx] */
    float* v_cache;  /* [n_layer, n_head_kv, head_dim, max_ctx] */
    float* hidden;   /* [n_embd]      — current residual stream */
    float* normed;   /* [n_embd]      — RMS norm output (temp)  */
    float* buf_q;    /* [n_embd]      — Q values, then attn out */
    float* buf_k;    /* [max(n_ff,E)] — K values / FFN up / gate */
    float* buf_v;    /* [n_embd]      — V values / FFN down out  */
    float* scores;   /* [max_ctx]     — attention scores        */
    float* ffbuf;    /* [n_ff]        — SiLU(gate) * up / res   */
    float* logits;   /* [n_vocab]     — output logits           */
    /* ── MLA (DeepSeek2) compressed KV cache (NULL for non-MLA models) ── */
    float* mla_kv_cache;     /* [n_layer][kv_lora_rank + qk_rope_head_dim][max_ctx] */

    /* ── SSM (Mamba) per-layer state caches (NULL for non-SSM models) ── */
    float* ssm_conv_state;   /* [n_layer][d_inner][d_conv-1]   */
    float* ssm_hidden_state; /* [n_layer][d_state][d_inner]    */
    int gpu_layers;   /* layers offloaded to GPU (0 = CPU only, 99 = all) */
    void* vk_backend; /* ct_vulkan_backend*, optional GPU offload */
} ct_infer_state;

/* ─── Internal functions exposed for diagnostics ─── */

void rope(float* buf, int n, int pos, float base);
void rms_norm(float* y, const float* x, const float* w, int n, float eps);
void matmul(float* y, const float* x, const void* w, int type, int I, int O);
/* Embedding lookup with vocab bounds check. Returns 0 on success, -1 if
 * token is out of range (out is zeroed). n_vocab = number of rows in table. */
int embed_row(float* out, const void* table, int type, int token, int n_embd, int n_vocab);

/* ─── API ─── */

/* Load model weights from an already-opened GGUF context.
 * max_ctx: override KV cache context length (0 = use GGUF default).
 * gpu_layers: number of layers to offload to GPU (0 = CPU only, 99 = all).
 * Returns NULL on failure (unsupported arch, missing tensors). */
ct_infer_state* ct_infer_create(ct_gguf_context* gguf, int max_ctx, int gpu_layers);

/* Run forward pass for position `pos` using hidden_in as the input state.
 * hidden_out receives the output (one transformer block).
 * pos is used for RoPE and KV caching. */
int ct_infer_forward(ct_infer_state* s, int pos,
                     const float* hidden_in, float* hidden_out);

/* Sample next token ID from logits.
 * top_k = 0 means no top-k filtering.
 * top_p = 1.0 means no nucleus filtering.
 * Returns token ID (0 = failure). */
int ct_infer_sample(const float* logits, int n_vocab, float temp, int top_k, float top_p);

/* Generate tokens autoregressively with optional per-token streaming callback.
 * tokens[0..n_prompt-1] are input token IDs.
 * Appends generated token IDs to output_tokens (up to max_gen).
 * If on_token is not NULL, called with each generated token ID (for SSE/streaming).
 * top_p = 1.0 = disabled. repeat_penalty = 1.0 = disabled. top_k = 0 = disabled.
 * Returns number of tokens generated, or <0 on error. */
int ct_infer_generate(ct_infer_state* s,
                      const int* tokens, int n_prompt,
                      int max_gen, float temp, int eos_id,
                      int* output_tokens,
                      void (*on_token)(int token, void* ctx),
                      void* stream_ctx,
                      float top_p, float repeat_penalty, int top_k);

/* Free all inference buffers. */
void ct_infer_free(ct_infer_state* s);

#ifdef __cplusplus
}
#endif

#endif /* CALM_INFER_H */

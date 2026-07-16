/**
 * calm_infer.h — Calm Native Inference Engine
 *
 * LLaMA-family transformer forward pass with quantized weights.
 * Supports F32, Q8_0, Q4_0, BQ1_0, TQ1_0 matmul dispatch.
 *
 * Usage:
 *   ct_gguf_context* gguf = ct_gguf_open("model.gguf");
 *   ct_infer_state* s = ct_infer_create(gguf);
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
    float norm_rms_eps;
    float rope_freq_base;
} ct_infer_config;

/* ─── Per-layer weight pointers into mmap ─── */
typedef struct {
    float* attn_norm;          /* F32 — norm weights always float */
    void*  attn_q;    int t_q;
    float* attn_q_bias;        /* Q bias (F32), may be NULL */
    void*  attn_k;    int t_k;
    float* attn_k_bias;        /* K bias (F32), may be NULL */
    void*  attn_v;    int t_v;
    float* attn_v_bias;        /* V bias (F32), may be NULL */
    void*  attn_out;  int t_o;
    float* ffn_norm;
    void*  ffn_gate;  int t_g;
    void*  ffn_up;    int t_u;
    void*  ffn_down;  int t_d;
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
} ct_infer_state;

/* ─── Internal functions exposed for diagnostics ─── */

void rope(float* buf, int n, int pos, float base);
void rms_norm(float* y, const float* x, const float* w, int n, float eps);
void matmul(float* y, const float* x, const void* w, int type, int I, int O);
void embed_row(float* out, const void* table, int type, int token, int n_embd);

/* ─── API ─── */

/* Load model weights from an already-opened GGUF context.
 * Returns NULL on failure (unsupported arch, missing tensors). */
ct_infer_state* ct_infer_create(ct_gguf_context* gguf);

/* Run forward pass for position `pos` using hidden_in as the input state.
 * hidden_out receives the output (one transformer block).
 * pos is used for RoPE and KV caching. */
int ct_infer_forward(ct_infer_state* s, int pos,
                     const float* hidden_in, float* hidden_out);

/* Sample next token ID from logits using temperature + argmax.
 * Returns token ID (0 = failure). */
int ct_infer_sample(const float* logits, int n_vocab, float temp);

/* Generate tokens autoregressively.
 * tokens[0..n_prompt-1] are input token IDs.
 * Appends generated token IDs to output_tokens (up to max_gen).
 * Returns number of tokens generated, or <0 on error. */
int ct_infer_generate(ct_infer_state* s,
                      const int* tokens, int n_prompt,
                      int max_gen, float temp, int eos_id,
                      int* output_tokens);

/* Free all inference buffers. */
void ct_infer_free(ct_infer_state* s);

#ifdef __cplusplus
}
#endif

#endif /* CALM_INFER_H */

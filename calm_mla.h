/**
 * calm_mla.h — Multi-head Latent Attention (DeepSeek2) Forward Pass
 *
 * MLA compresses K/V into a low-rank latent space, dramatically reducing
 * KV cache size (e.g. 512 vs 4096 floats per token for DS-Coder-V2-Lite).
 *
 * Combined with Decoupled RoPE for positional encoding.
 */
#ifndef CALM_MLA_H
#define CALM_MLA_H

#include "calm_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* MLA forward pass: compute attention using compressed latent K/V.
 *
 * normed  — RMS-normalized input [n_embd]
 * lw      — per-layer weights (MLA fields)
 * cfg     — model config with MLA dimensions
 * k_cache — per-layer MLA KV cache [kv_lora_rank + qk_rope_head_dim, max_ctx]
 * pos     — current token position
 * max_ctx — KV cache capacity
 * buf_q   — output buffer [n_head * v_head_dim], then used as attn_out
 */
void ct_forward_mla(float* buf_q, const float* normed,
                    const ct_infer_layer* lw,
                    const ct_infer_config* cfg,
                    float* k_cache, int pos, int max_ctx);

#ifdef __cplusplus
}
#endif

#endif /* CALM_MLA_H */

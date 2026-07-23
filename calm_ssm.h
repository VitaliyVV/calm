/**
 * calm_ssm.h — Calm State Space Model Forward Pass
 *
 * Mamba1-style selective scan with 1D convolution + discretized recurrence.
 * Used by hybrid architectures: Ornith, Qwythos (Qwen3.5 SSM), Jamba.
 *
 * Token-by-token inference: conv state + SSM hidden state cached per-layer.
 */
#ifndef CALM_SSM_H
#define CALM_SSM_H

#include "calm_infer.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── SSM Convolution (depthwise 1D, token-by-token) ───
 *
 * conv_state[d_inner][d_conv-1] : FIFO buffer of previous inputs.
 * For each channel i:
 *   out[i] = w[0]*x[i] + sum_{k=1}^{d_conv-1} w[k]*state[i][k-1]
 * Then shift: state[i][0..d_conv-3] = state[i][1..d_conv-2],
 *             state[i][d_conv-2] = x[i]   */
void ct_ssm_conv1d(float* out, const float* x,
                   const void* weight, int t_weight,
                   const float* bias,
                   float* conv_state,
                   int d_conv, int d_inner);

/* ─── SSM Selective Scan (Mamba1, single-token) ───
 *
 * The core recurrence: for each channel i in 0..d_inner-1:
 *   dt_i     = expf(dt[i])
 *   A_bar    = expf(dt_i * A_col[s][i])           // discretized A
 *   B_bar    = dt_i * B[s]                         // discretized B
 *   h_new[s] = A_bar * h_cache[s][i] + B_bar * x[i]
 *   y[i]    += C[s] * h_new[s]
 *   y[i]    += D[i] * x[i]
 *
 * A: [d_state, d_inner] in GGUF storage order (weight has this shape)
 * h_cache: [d_state, d_inner] flat (interleaved by channel)   */
void ct_ssm_selective_scan(float* y, const float* x,
                           const float* dt,
                           const float* B, const float* C,
                           const void* A, int t_A,
                           const void* D, int t_D,
                           float* h_cache,
                           int d_state, int d_inner);

/* ─── Full SSM Block Forward (Mamba1) ───
 *
 * Implements one Mamba1 block for a single token position.
 * Reads/writes conv_state[d_inner][d_conv-1] and h_state[d_state][d_inner].
 *
 * Weight pointers and types are from ct_infer_layer.ssm_* fields.
 * out: [n_embd] — output to add to residual stream.      */
void ct_forward_ssm(float* out, const float* hidden_in,
                    const ct_infer_layer* lw,
                    const ct_infer_config* cfg,
                    float* conv_state,
                    float* h_state);

/* ─── Full SSM Block Forward (Qwythos/Qwen3.5 variant) ───
 *
 * Implements the Qwythos SSM variant for a single token position.
 * This variant uses:
 *   - attn_qkv.weight as fused input projection [n_embd, 2*d_inner]
 *   - ssm_alpha.weight for dt/B/C projection [n_embd, d_state]
 *   - ssm_beta.weight for output gate [n_embd, d_state]
 *   - ssm_a [d_state] as diagonal A (not [d_state, d_inner])
 *   - No ssm_d (skip connection)
 *   - No separate ssm_x, ssm_dt.weight (dt from bias + alpha only)
 *
 * conv_state is [2*d_inner][d_conv-1] (conv1d on full xz).
 * h_state is [d_state][d_inner] (d_inner = n_embd, not 2*n_embd).
 *
 * out: [n_embd] — output to add to residual stream.      */
void ct_forward_ssm_qwythos(float* out, const float* hidden_in,
                            const ct_infer_layer* lw,
                            const ct_infer_config* cfg,
                            float* conv_state,
                            float* h_state);

#ifdef __cplusplus
}
#endif

#endif /* CALM_SSM_H */

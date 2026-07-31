/**
 * calm_ssm.c — Calm State Space Model Forward Pass
 *
 * Mamba1 selective scan + 1D convolution for token-by-token inference.
 * All matmuls dispatch through the existing quantized matmul() in calm_infer.c.
 */
#include "calm_ssm.h"
#include "calm_quant.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════
 * Internal: dequantize one contiguous row of weight into float buf.
 *
 * Handles common quant types for small-rows (SSM conv1d: d_conv <= 4).
 * Row data starts at (const uint8_t*)weight + row_offset.
 * n = number of elements in the row (d_inner).
 * ═══════════════════════════════════════════════════════════════ */
static void dequant_row(float* buf, const void* base, int type, int row_offset, int n) {
    const uint8_t* row = (const uint8_t*)base + (size_t)row_offset;

    switch (type) {
        case CT_GGUF_TYPE_F32: {
            const float* src = (const float*)row;
            for (int i = 0; i < n; i++) buf[i] = src[i];
            break;
        }
        case CT_GGUF_TYPE_F16: {
            const uint16_t* src = (const uint16_t*)row;
            for (int i = 0; i < n; i++) buf[i] = ct_fp16_to_fp32(src[i]);
            break;
        }
        case CT_GGUF_TYPE_Q8_0: {
            const ct_block_q8_0* blk = (const ct_block_q8_0*)row;
            int nb = (n + 31) / 32;
            for (int b = 0; b < nb; b++) {
                float d = ct_fp16_to_fp32(blk[b].d);
                int rem = n - b * 32;
                int lim = rem < 32 ? rem : 32;
                for (int i = 0; i < lim; i++)
                    buf[b * 32 + i] = blk[b].qs[i] * d;
            }
            break;
        }
        case CT_GGUF_TYPE_Q4_0: {
            const ct_block_q4_0* blk = (const ct_block_q4_0*)row;
            int nb = (n + 31) / 32;
            for (int b = 0; b < nb; b++) {
                float d = ct_fp16_to_fp32(blk[b].d);
                int rem = n - b * 32;
                int lim = rem < 32 ? rem : 32;
                for (int i = 0; i < lim; i++) {
                    int nib = (blk[b].qs[i >> 1] >> ((i & 1) << 2)) & 0xF;
                    buf[b * 32 + i] = ((float)nib - 8.0f) * d;
                }
            }
            break;
        }
        default:
            /* Unsupported type — zero the row */
            fprintf(stderr, "ssm: unsupported conv1d quant type %d\n", type);
            memset(buf, 0, (size_t)n * sizeof(float));
            break;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * ssm_conv1d — Depthwise 1D Convolution (single token)
 *
 * conv_state[d_inner][d_conv-1] stored row-major:
 *   state[i * (d_conv-1) + k] = input from (k+1) positions ago for channel i
 *
 * For each channel i:
 *   out[i] = W[0][i]*x[i] + sum_{k=1}^{d_conv-1} W[k][i]*state[i][k-1]
 *
 * Then shift state and append x[i].
 *
 * Weight has shape [d_conv, d_inner] in GGUF (row-major: d_conv rows).
 * Row k of weight is at byte offset k * row_stride.
 * ═══════════════════════════════════════════════════════════════ */
void ct_ssm_conv1d(float* out, const float* x,
                   const void* weight, int t_weight,
                   const float* bias,
                   float* conv_state,
                   int d_conv, int d_inner) {
    if (d_conv < 1) return;

    /* Temporary buffer for dequantized weight row */
    float* w_row = (float*)malloc((size_t)d_inner * sizeof(float));
    if (!w_row) {
        fprintf(stderr, "ssm: OOM in conv1d (d_inner=%d)\n", d_inner);
        memset(out, 0, (size_t)d_inner * sizeof(float));
        return;
    }

    /* Compute row stride in bytes for this quant type */
    size_t row_stride;
    uint64_t dims[2] = {(uint64_t)d_inner, 1};
    row_stride = ct_gguf_tensor_size(t_weight, 2, dims);

    /* Initialize output with bias */
    if (bias) {
        for (int i = 0; i < d_inner; i++) out[i] = bias[i];
    } else {
        memset(out, 0, (size_t)d_inner * sizeof(float));
    }

    /* k=0: current token x[i]  */
    dequant_row(w_row, weight, t_weight, 0, d_inner);
    for (int i = 0; i < d_inner; i++)
        out[i] += w_row[i] * x[i];

    /* k=1..d_conv-1: previous tokens from conv_state */
    int c_stride = d_conv - 1;
    for (int k = 1; k < d_conv; k++) {
        dequant_row(w_row, weight, t_weight, (int)k * (int)row_stride, d_inner);
        for (int i = 0; i < d_inner; i++)
            out[i] += w_row[i] * conv_state[i * c_stride + (k - 1)];
    }

    /* Shift conv_state: drop oldest (col d_conv-2), shift left, append x */
    for (int i = 0; i < d_inner; i++) {
        /* shift */
        for (int k = 0; k < c_stride - 1; k++)
            conv_state[i * c_stride + k] = conv_state[i * c_stride + k + 1];
        /* append new value */
        if (c_stride > 0)
            conv_state[i * c_stride + (c_stride - 1)] = x[i];
    }

    free(w_row);
}

/* ═══════════════════════════════════════════════════════════════
 * ssm_selective_scan — Mamba1 recurrent core (single token)
 *
 * Implements the discretized SSM recurrence for one time step:
 *
 * For each channel i in 0..d_inner-1:
 *   Δ_i     = expf(dt[i])              // discretization step
 *   For s in 0..d_state-1:
 *     A_bar = expf(Δ_i * A[s][i])
 *     h_new[s][i] = A_bar * h_cache[s][i] + Δ_i * B[s] * x[i]
 *   y[i] = dot(C, h_new[:, i]) + D[i] * x[i]
 *
 * A layout: GGUF stores [d_state, d_inner] — i.e. A[s][i] at s*d_inner + i.
 * h_cache layout: [d_state, d_inner] channel-major — h[s][i] at s*d_inner + i.
 * ═══════════════════════════════════════════════════════════════ */
void ct_ssm_selective_scan(float* y, const float* x,
                           const float* dt,
                           const float* B, const float* C,
                           const void* A, int t_A,
                           const void* D, int t_D,
                           float* h_cache,
                           int d_state, int d_inner) {
    float* h_new = (float*)malloc((size_t)d_state * d_inner * sizeof(float));
    if (!h_new) {
        fprintf(stderr, "ssm: OOM in selective_scan (S=%d, I=%d)\n", d_state, d_inner);
        memset(y, 0, (size_t)d_inner * sizeof(float));
        return;
    }

    /* Temporary buffer for dequantized A column slice (d_state per channel).
     * A is [d_state, d_inner] — A[s][i] at s*d_inner + i.
     * For each channel i, A[:, i] is strided by d_inner — not contiguous!
     * So we dequantize channel-by-channel. */
    float* h_col  = (float*)malloc((size_t)d_state * sizeof(float));
    float* a_col  = (float*)malloc((size_t)d_state * sizeof(float));
    float* d_val  = (float*)malloc(sizeof(float));
    if (!h_col || !a_col || !d_val) {
        fprintf(stderr, "ssm: OOM in selective_scan\n");
        memset(y, 0, (size_t)d_inner * sizeof(float));
        free(h_new); free(h_col); free(a_col); free(d_val);
        return;
    }

    /* Scratch for element-wise matmul replacement (single d_state column) */
    float* deq_scratch = (float*)malloc((size_t)d_state * sizeof(float));
    if (!deq_scratch) {
        fprintf(stderr, "ssm: OOM in selective_scan\n");
        memset(y, 0, (size_t)d_inner * sizeof(float));
        free(h_new); free(h_col); free(a_col); free(d_val);
        return;
    }

    for (int i = 0; i < d_inner; i++) {
        float xi = x[i];
        /* Numerically stable softplus: Δ = log(1 + exp(dt))
         * For dt > 20: log(1 + exp(dt)) ≈ dt (avoid expf overflow)
         * For dt < -20: log(1 + exp(dt)) ≈ exp(dt) ≈ 0
         * Mamba1 always uses softplus for the discretization step Δ. */
        float dti;
        if (dt[i] > 20.0f) {
            dti = dt[i];  /* log(1+exp(dt)) ≈ dt for large dt */
        } else {
            dti = logf(1.0f + expf(dt[i]));
        }

        /* Dequantize A[:, i] — elements are at A[s * d_inner + i] for s=0..d_state-1.
         * Since d_state is typically small (16-128), do element-wise dequant. */
        {
            /* Find byte offset of A[0][i] = A[0*d_inner + i] */
            /* Then read strided by d_inner elements = d_inner * element_size_in_file */

            /* For each state s: A[s][i] = base + s*inner_element_stride + i*element_size_in_type */
            /* But the actual in-file format is blocks... */
            /* This is the tricky part: we need random access into a quantized tensor. */

            /* For now, only handle the types where single-element access is feasible.
             * F32 and F16 are trivial. Q8_0 and Q4_0 need block-based access. */
            switch (t_A) {
                case CT_GGUF_TYPE_F32: {
                    const float* Af32 = (const float*)A;
                    for (int s = 0; s < d_state; s++)
                        a_col[s] = Af32[s * d_inner + i];
                    break;
                }
                case CT_GGUF_TYPE_F16: {
                    const uint16_t* Af16 = (const uint16_t*)A;
                    for (int s = 0; s < d_state; s++)
                        a_col[s] = ct_fp16_to_fp32(Af16[s * d_inner + i]);
                    break;
                }
                case CT_GGUF_TYPE_Q8_0: {
                    /* Q8_0 blocks have 32 elements. A[s][i] is at block (s*d_inner+i)/32,
                     * offset (s*d_inner+i)%32 within the block. */
                    const ct_block_q8_0* blk = (const ct_block_q8_0*)A;
                    for (int s = 0; s < d_state; s++) {
                        int idx = s * d_inner + i;
                        int bidx = idx / 32;
                        int off = idx % 32;
                        float d = ct_fp16_to_fp32(blk[bidx].d);
                        a_col[s] = blk[bidx].qs[off] * d;
                    }
                    break;
                }
                case CT_GGUF_TYPE_Q4_0: {
                    /* Q4_0: nibble-packed */
                    const ct_block_q4_0* blk = (const ct_block_q4_0*)A;
                    for (int s = 0; s < d_state; s++) {
                        int idx = s * d_inner + i;
                        int bidx = idx / 32;
                        int off = idx % 32;
                        float d = ct_fp16_to_fp32(blk[bidx].d);
                        int nib = (blk[bidx].qs[off >> 1] >> ((off & 1) << 2)) & 0xF;
                        a_col[s] = ((float)nib - 8.0f) * d;
                    }
                    break;
                }
                default: {
                    /* Unsupported — force zero */
                    fprintf(stderr, "ssm: unsupported A quant type %d at channel %d\n", t_A, i);
                    for (int s = 0; s < d_state; s++) a_col[s] = 0.0f;
                    break;
                }
            }
        }

        /* Dequantize D[i] — D is [d_inner] */
        float d_i = 0.0f;
        {
            switch (t_D) {
                case CT_GGUF_TYPE_F32:
                    d_i = ((const float*)D)[i];
                    break;
                case CT_GGUF_TYPE_F16:
                    d_i = ct_fp16_to_fp32(((const uint16_t*)D)[i]);
                    break;
                case CT_GGUF_TYPE_Q8_0: {
                    const ct_block_q8_0* blk = (const ct_block_q8_0*)D;
                    int bidx = i / 32, off = i % 32;
                    d_i = ct_fp16_to_fp32(blk[bidx].d) * blk[bidx].qs[off];
                    break;
                }
                default:
                    d_i = 0.0f;
                    break;
            }
        }

        /* Selective scan for this channel:
         *   h_new[s] = expf(dti * A[s][i]) * h_cache[s][i] + dti * B[s] * xi
         *   y[i] = sum_s C[s] * h_new[s] + D[i] * xi
         */
        float yi = xi * d_i;  /* D contribution */
        for (int s = 0; s < d_state; s++) {
            float A_bar = expf(dti * a_col[s]);
            float B_bar = dti * B[s];
            float h_old = h_cache[s * d_inner + i];
            float hn = A_bar * h_old + B_bar * xi;
            h_new[s * d_inner + i] = hn;
            yi += C[s] * hn;
        }
        y[i] = yi;
    }

    /* Copy h_new → h_cache for next token */
    memcpy(h_cache, h_new, (size_t)d_state * d_inner * sizeof(float));

    free(h_new);
    free(h_col);
    free(a_col);
    free(d_val);
    free(deq_scratch);
}

/* ═══════════════════════════════════════════════════════════════
 * forward_ssm — Full Mamba1 Block (one token position)
 *
 * Weight pointers/types come from ct_infer_layer fields.
 * conv_state/h_state are per-layer persistent caches.
 *
 * Pipeline:
 *   1. xz = ssm_in @ hidden_in          (input projection + gate)
 *   2. x, z = split(xz)                 (gate/state split)
 *   3. conv_out = conv1d(x) + bias      (depthwise 1D conv)
 *   4. conv_out = silu(conv_out)        (activation)
 *   5. x_db = ssm_x @ conv_out          (dt, B, C projection)
 *   6. dt_raw, B, C = split(x_db)       (split)
 *   7. Optional RMS norm: dt, B, C      (Jamba-style)
 *   8. dt = ssm_dt @ dt_raw + dt_bias   (time step projection)
 *   9. y = selective_scan(conv_out, dt, B, C, A, D, h_state)
 *  10. y += D * conv_out                (skip connection — done in scan)
 *  11. y = silu(z) * y                  (output gating)
 *  12. out = ssm_out @ y                (output projection)
 *
 * out: [n_embd]
 * ═══════════════════════════════════════════════════════════════ */
void ct_forward_ssm(float* out, const float* hidden_in,
                    const ct_infer_layer* lw,
                    const ct_infer_config* cfg,
                    float* conv_state,
                    float* h_state) {
    int E = cfg->n_embd;
    int d_inner = cfg->ssm_d_inner;
    int d_conv  = cfg->ssm_d_conv;
    int d_state = cfg->ssm_d_state;
    int dt_rank = cfg->ssm_dt_rank;

    if (d_inner == 0 || d_conv == 0 || d_state == 0) {
        fprintf(stderr, "ssm: invalid SSM config (I=%d C=%d S=%d)\n",
                d_inner, d_conv, d_state);
        memset(out, 0, (size_t)E * sizeof(float));
        return;
    }

    /* Use scratch buffers from ct_infer_state.
     * We need: xz[2*d_inner], dt_raw[dt_rank], B[d_state], C[d_state],
     *          dt_proj[d_inner], conv_out[d_inner], scan_out[d_inner],
     *          silu_z[d_inner], gated_y[d_inner] */

    /* Allocate on stack for small sizes, heap otherwise */
    float* xz      = (float*)malloc((size_t)(2 * d_inner) * sizeof(float));
    float* dt_raw  = (float*)malloc((size_t)(dt_rank > 0 ? dt_rank : 1) * sizeof(float));
    float* B       = (float*)malloc((size_t)d_state * sizeof(float));
    float* C       = (float*)malloc((size_t)d_state * sizeof(float));
    float* dt_proj = (float*)malloc((size_t)d_inner * sizeof(float));
    float* conv_x  = (float*)malloc((size_t)d_inner * sizeof(float));
    float* tmp     = (float*)malloc((size_t)(d_inner > E ? d_inner : E) * sizeof(float));

    if (!xz || !dt_raw || !B || !C || !dt_proj || !conv_x || !tmp) {
        fprintf(stderr, "ssm: OOM in forward_ssm (I=%d, S=%d, R=%d, E=%d)\n",
                d_inner, d_state, dt_rank, E);
        memset(out, 0, (size_t)E * sizeof(float));
        free(xz); free(dt_raw); free(B); free(C);
        free(dt_proj); free(conv_x); free(tmp);
        return;
    }

    /* Step 1-2: xz = ssm_in @ hidden_in → split into x (first d_inner) and z */
    matmul(xz, hidden_in, lw->ssm_in, lw->t_ssm_in, E, 2 * d_inner);
    float* x_part = xz;           /* first d_inner elements */
    float* z_part = xz + d_inner; /* last d_inner elements  */

    /* Step 3-4: conv_out = silu(conv1d(x) + bias) */
    ct_ssm_conv1d(conv_x, x_part, lw->ssm_conv1d, lw->t_ssm_conv1d,
                  lw->ssm_conv1d_b, conv_state, d_conv, d_inner);
    for (int i = 0; i < d_inner; i++)
        conv_x[i] = conv_x[i] / (1.0f + expf(-conv_x[i])); /* silu */

    /* Step 5: x_db = ssm_x @ conv_x → dt_raw, B, C */
    matmul(tmp, conv_x, lw->ssm_x, lw->t_ssm_x, d_inner, dt_rank + 2 * d_state);
    memcpy(dt_raw, tmp, (size_t)dt_rank * sizeof(float));
    memcpy(B,      tmp + dt_rank, (size_t)d_state * sizeof(float));
    memcpy(C,      tmp + dt_rank + d_state, (size_t)d_state * sizeof(float));

    /* Step 7: Optional RMS norm on dt, B, C (Jamba-style) */
    if (cfg->ssm_dt_b_c_rms && lw->ssm_dt_norm && lw->ssm_b_norm && lw->ssm_c_norm) {
        rms_norm(dt_raw, dt_raw, lw->ssm_dt_norm, dt_rank, cfg->norm_rms_eps);
        rms_norm(B, B, lw->ssm_b_norm, d_state, cfg->norm_rms_eps);
        rms_norm(C, C, lw->ssm_c_norm, d_state, cfg->norm_rms_eps);
    }

    /* Step 8: dt = ssm_dt @ dt_raw + dt_bias */
    matmul(dt_proj, dt_raw, lw->ssm_dt, lw->t_ssm_dt, dt_rank, d_inner);
    if (lw->ssm_dt_b) {
        for (int i = 0; i < d_inner; i++)
            dt_proj[i] += lw->ssm_dt_b[i];
    }
    /* Step 9-10: y = selective_scan(conv_x, dt_proj, B, C, A, D, h_state) */
    ct_ssm_selective_scan(tmp, conv_x, dt_proj, B, C,
                          lw->ssm_a, lw->t_ssm_a,
                          lw->ssm_d, lw->t_ssm_d,
                          h_state, d_state, d_inner);

    /* Step 11: y = silu(z) * y   (output gating) */
    for (int i = 0; i < d_inner; i++) {
        float sig = 1.0f / (1.0f + expf(-z_part[i]));
        tmp[i] *= sig;
    }

    /* Step 12: out = ssm_out @ y */
    matmul(out, tmp, lw->ssm_out, lw->t_ssm_out, d_inner, E);

    free(xz);
    free(dt_raw);
    free(B);
    free(C);
    free(dt_proj);
    free(conv_x);
    free(tmp);
}

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
        case CT_GGUF_TYPE_BQ1_0: {
            /* BQ1_0: 128 elements per block, 2× uint64 bits + FP16 scale.
             * bit=1 → +d, bit=0 → -d */
            const ct_block_bq1_0* blk = (const ct_block_bq1_0*)row;
            int nb = (n + 127) / 128;
            for (int b = 0; b < nb; b++) {
                float d = ct_fp16_to_fp32(blk[b].d);
                int rem = n - b * 128;
                int lim = rem < 128 ? rem : 128;
                for (int i = 0; i < lim; i++) {
                    int byte_idx = i >> 6;       /* i / 64 */
                    int bit_idx  = i & 0x3F;      /* i % 64 */
                    int bit      = (blk[b].bits[byte_idx] >> bit_idx) & 1;
                    buf[b * 128 + i] = bit ? d : -d;
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
            uint64_t dims[2] = {(uint64_t)d_inner, (uint64_t)d_state};
            size_t tensor_size = ct_gguf_tensor_size(t_A, 2, dims);

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

/* ═══════════════════════════════════════════════════════════════════
 * ct_forward_ssm_qwythos — Qwythos/Qwen3.5 SSM variant
 *
 * Architecture differences from Mamba1:
 *   - Input projection via attn_qkv.weight (not ssm_in)
 *   - ssm_alpha projects normed hidden → dt/B/C in one go (d_state dims)
 *   - ssm_beta produces per-state output gate from normed hidden
 *   - Diagonal A [d_state] (not [d_state, d_inner])
 *   - No ssm_x, no ssm_dt.weight, no ssm_d (skip connection)
 *   - dt is per-state (d_state), not per-channel (d_inner)
 *   - conv1d operates on the full xz (2*d_inner channels)
 *
 * conv_state: [2*d_inner][d_conv-1] per layer
 * h_state:    [d_state][d_inner] per layer
 *
 * The caller (ct_infer_forward) is responsible for computing the
 * per-layer pointer offsets. d_inner here is n_embd (not 2*n_embd).
 * ═══════════════════════════════════════════════════════════════════ */
void ct_forward_ssm_qwythos(float* out, const float* hidden_in,
                            const ct_infer_layer* lw,
                            const ct_infer_config* cfg,
                            float* conv_state,
                            float* h_state) {
    int E = cfg->n_embd;           /* 4096 */
    int d_inner = E;               /* scan dimension (x, not xz) */
    int d_xz    = 2 * d_inner;     /* 8192 — conv1d on full xz */
    int d_conv  = cfg->ssm_d_conv; /* 4    */
    int d_state = cfg->ssm_d_state;/* 128  */

    /* Qwythos/qwen35 grouped SSM:
     * d_state = total states, group_count = parameter groups.
     * Each group shares A/dt/alpha/beta for state_per_group sub-states. */
    int n_groups = cfg->ssm_group_count > 0 ? cfg->ssm_group_count : cfg->ssm_dt_rank;
    int state_per_group = (n_groups > 0) ? d_state / n_groups : d_state;

    if (d_inner == 0 || d_conv == 0 || d_state == 0 || n_groups <= 0) {
        fprintf(stderr, "ssm_qwythos: invalid config (I=%d C=%d S=%d G=%d)\n",
                d_inner, d_conv, d_state, n_groups);
        memset(out, 0, (size_t)E * sizeof(float));
        return;
    }

    /* Allocate scratch buffers — alpha/beta/dt sized per group (n_groups), not per state */
    float* xz      = (float*)malloc((size_t)d_xz * sizeof(float));
    float* xz_conv = (float*)malloc((size_t)d_xz * sizeof(float));
    float* alpha   = (float*)malloc((size_t)n_groups * sizeof(float));
    float* beta    = (float*)malloc((size_t)n_groups * sizeof(float));
    float* dt      = (float*)malloc((size_t)n_groups * sizeof(float));
    float* y       = (float*)malloc((size_t)d_inner * sizeof(float));
    float* tmp     = (float*)malloc((size_t)E * sizeof(float));

    if (!xz || !xz_conv || !alpha || !beta || !dt || !y || !tmp) {
        fprintf(stderr, "ssm_qwythos: OOM\n");
        goto cleanup;
    }

    /* Step 1: xz = attn_qkv @ hidden_in → [2 * d_inner] */
    matmul(xz, hidden_in, lw->ssm_qkv, lw->t_ssm_qkv, E, d_xz);

    /* Step 2: conv_xz = silu(conv1d(xz) + bias) on all d_xz channels */
    ct_ssm_conv1d(xz_conv, xz, lw->ssm_conv1d, lw->t_ssm_conv1d,
                  lw->ssm_conv1d_b, conv_state, d_conv, d_xz);
    for (int i = 0; i < d_xz; i++)
        xz_conv[i] = xz_conv[i] / (1.0f + expf(-xz_conv[i])); /* silu */

    /* Step 3: Split into x (first d_inner) and z (last d_inner) */
    float* x_part = xz_conv;
    float* z_part = xz_conv + d_inner;

    /* Step 4: alpha = ssm_alpha @ normed_input → [n_groups]
     *   ssm_alpha.weight is [n_embd, n_groups] = [4096, 32] */
    matmul(alpha, hidden_in, lw->ssm_alpha, lw->t_ssm_alpha, E, n_groups);

    /* Step 5: dt = exp(alpha + ssm_dt_bias) — per-group discretization step */
    if (lw->ssm_dt_b) {
        for (int g = 0; g < n_groups; g++)
            dt[g] = expf(alpha[g] + lw->ssm_dt_b[g]);
    } else {
        for (int g = 0; g < n_groups; g++)
            dt[g] = expf(alpha[g]);
    }

    /* Step 6: beta_gate = sigmoid(ssm_beta @ normed_input)
     *   ssm_beta.weight is [n_embd, n_groups] = [4096, 32] */
    matmul(beta, hidden_in, lw->ssm_beta, lw->t_ssm_beta, E, n_groups);
    for (int g = 0; g < n_groups; g++)
        beta[g] = 1.0f / (1.0f + expf(-beta[g]));

    /* Step 7: Grouped selective scan
     *
     * For each group g (0..n_groups-1):
     *   dt_g    = dt[g]                              (discretization step)
     *   a_bar   = exp(dt_g * A[g])                   (diagonal A per group)
     *   b_bar   = dt_g * alpha[g]                    (B = alpha, discretized)
     *   c_gated = alpha[g] * beta[g]                 (C * output gate)
     *
     *   For each sub-state s in group (0..state_per_group-1):
     *     global_state = g * state_per_group + s
     *     h_new[gs][i] = a_bar * h_old[gs][i] + b_bar * x_part[i]
     *     y[i] += c_gated * h_new[gs][i]
     *
     * h_state layout: [d_state][d_inner] contiguous per layer.
     * A layout:       [n_groups] diagonal elements (BQ1_0 or F32).
     */
    {
        /* Dequantize diagonal A (BQ1_0 → float) */
        float* A_vals = (float*)malloc((size_t)n_groups * sizeof(float));
        if (!A_vals) {
            fprintf(stderr, "ssm_qwythos: OOM for A\n");
            goto cleanup;
        }
        dequant_row(A_vals, lw->ssm_a, lw->t_ssm_a, 0, n_groups);

        /* Initialize y to zero */
        memset(y, 0, (size_t)d_inner * sizeof(float));

        for (int g = 0; g < n_groups; g++) {
            float dt_g      = dt[g];
            float a_bar_g   = expf(dt_g * A_vals[g]);
            float b_bar_g   = dt_g * alpha[g];
            float c_gated_g = alpha[g] * beta[g];

            /* Apply to all sub-states within this group */
            for (int sub = 0; sub < state_per_group; sub++) {
                int s = g * state_per_group + sub;
                float* h_s = h_state + (size_t)s * d_inner;

                for (int i = 0; i < d_inner; i++) {
                    float h_new = a_bar_g * h_s[i] + b_bar_g * x_part[i];
                    h_s[i] = h_new;
                    y[i] += c_gated_g * h_new;
                }
            }
        }
        free(A_vals);
    }

    /* Step 8: y[i] *= sigmoid(z[i]) — output gate from z */
    for (int i = 0; i < d_inner; i++) {
        float sig = 1.0f / (1.0f + expf(-z_part[i]));
        y[i] *= sig;
    }

    /* Step 9: out = ssm_out @ y → [E] */
    matmul(out, y, lw->ssm_out, lw->t_ssm_out, d_inner, E);

cleanup:
    free(xz);
    free(xz_conv);
    free(alpha);
    free(beta);
    free(dt);
    free(y);
    free(tmp);
}

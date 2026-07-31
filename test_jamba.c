/**
 * test_jamba.c — SSM test harness for Jamba (Mamba1) models
 *
 * Compile:
 *   clang -O0 -g -std=c11 -DCT_NEON -o test_jamba test_jamba.c \
 *         calm_ssm.c calm_infer.c calm_gguf.c calm_quant.c calm_tokenizer.c \
 *         -lm -lpthread
 *
 * Usage:
 *   ./test_jamba jamba-3b-iq2_m.gguf [n_tokens]
 */
#include "calm_infer.h"
#include "calm_ssm.h"
#include "calm_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void print_config(const ct_infer_config* cfg) {
    printf("=== Model Config ===\n");
    printf("  n_layer=%d  n_embd=%d  n_head=%d  n_head_kv=%d\n",
           cfg->n_layer, cfg->n_embd, cfg->n_head, cfg->n_head_kv);
    printf("  n_ff=%d  n_vocab=%d  head_dim=%d\n",
           cfg->n_ff, cfg->n_vocab, cfg->head_dim);
    printf("  n_ctx_max=%d  norm_rms_eps=%g  rope_freq_base=%.0f\n",
           cfg->n_ctx_max, cfg->norm_rms_eps, cfg->rope_freq_base);
    printf("  n_expert=%d  n_expert_per_token=%d\n",
           cfg->n_expert, cfg->n_expert_per_token);
    printf("  SSM: d_conv=%d  d_inner=%d  d_state=%d  dt_rank=%d  dt_b_c_rms=%d\n",
           cfg->ssm_d_conv, cfg->ssm_d_inner, cfg->ssm_d_state,
           cfg->ssm_dt_rank, cfg->ssm_dt_b_c_rms);
}

static void print_logit_stats(const float* logits, int n, const char* label) {
    float sum = 0, sum2 = 0, min = 1e38f, max = -1e38f;
    int nnan = 0, ninf = 0;
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (isnan(v)) { nnan++; continue; }
        if (isinf(v)) { ninf++; continue; }
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sum2 += v * v;
    }
    float mean = sum / (n - nnan - ninf);
    float std = sqrtf(sum2 / (n - nnan - ninf) - mean * mean);
    printf("  %s: mean=%.4f std=%.4f [%.4f, %.4f] nan=%d inf=%d\n",
           label, mean, std, min, max, nnan, ninf);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [n_tokens]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    int n_gen = (argc > 2) ? atoi(argv[2]) : 8;
    if (n_gen < 1) n_gen = 1;
    if (n_gen > 64) n_gen = 64;

    srand((unsigned)time(NULL));

    /* ── Open GGUF ── */
    printf("Opening %s...\n", model_path);
    ct_gguf_context* gguf = ct_gguf_open(model_path);
    if (!gguf) {
        fprintf(stderr, "FAIL: ct_gguf_open returned NULL\n");
        return 1;
    }
    printf("  architecture: %s\n", ct_gguf_architecture(gguf));

    /* ── Create inference state ── */
    printf("Creating inference state...\n");
    ct_infer_state* s = ct_infer_create(gguf, 512, 0);
    if (!s) {
        fprintf(stderr, "FAIL: ct_infer_create returned NULL\n");
        ct_gguf_close(gguf);
        return 1;
    }

    ct_infer_config* cfg = &s->w.config;
    print_config(cfg);

    /* ── Print per-layer type ── */
    printf("\n=== Layer Type Summary ===\n");
    int n_ssm = 0, n_attn = 0;
    for (int i = 0; i < cfg->n_layer; i++) {
        if (s->w.layers[i].is_ssm) {
            printf("  layer %3d: SSM (Mamba1)\n", i);
            n_ssm++;
        } else {
            printf("  layer %3d: Attention\n", i);
            n_attn++;
        }
    }
    printf("  Total: %d SSM + %d Attention = %d layers\n", n_ssm, n_attn, cfg->n_layer);

    /* ── Debug: look up ssm_in tensor dimensions ── */
    {
        const ct_gguf_tensor_info* ti = ct_gguf_find_tensor(gguf, "blk.0.ssm_in.weight");
        if (ti) {
            printf("\n=== Tensor Dims (blk.0.ssm_in.weight) ===\n");
            printf("  n_dims=%u  dims=[", ti->n_dims);
            for (int d = 0; d < (int)ti->n_dims; d++)
                printf("%s%llu", d ? "," : "", (unsigned long long)ti->dims[d]);
            printf("]  type=%d  size=%zu\n", ti->type, ti->size);
        }
    }

    /* ── Check SSM weight types ── */
    if (n_ssm > 0) {
        printf("\n=== SSM Weight Types (layer 0) ===\n");
        ct_infer_layer* l0 = &s->w.layers[0];
        printf("  ssm_in:      type=%d  ptr=%p\n", l0->t_ssm_in, l0->ssm_in);
        printf("  ssm_conv1d:  type=%d  ptr=%p\n", l0->t_ssm_conv1d, l0->ssm_conv1d);
        printf("  ssm_conv1d_b: ptr=%p\n", l0->ssm_conv1d_b);
        printf("  ssm_x:       type=%d  ptr=%p\n", l0->t_ssm_x, l0->ssm_x);
        printf("  ssm_dt:      type=%d  ptr=%p\n", l0->t_ssm_dt, l0->ssm_dt);
        printf("  ssm_dt_b:    ptr=%p\n", l0->ssm_dt_b);
        printf("  ssm_a:       type=%d  ptr=%p\n", l0->t_ssm_a, l0->ssm_a);
        printf("  ssm_d:       type=%d  ptr=%p\n", l0->t_ssm_d, l0->ssm_d);
        printf("  ssm_out:     type=%d  ptr=%p\n", l0->t_ssm_out, l0->ssm_out);
        if (l0->ssm_dt_norm) printf("  ssm_dt_norm:  ptr=%p\n", l0->ssm_dt_norm);
        if (l0->ssm_b_norm)  printf("  ssm_b_norm:   ptr=%p\n", l0->ssm_b_norm);
        if (l0->ssm_c_norm)  printf("  ssm_c_norm:   ptr=%p\n", l0->ssm_c_norm);
    }

    /* ── SSM caches ── */
    printf("\n=== SSM Caches ===\n");
    if (s->ssm_conv_state)
        printf("  ssm_conv_state:   ptr=%p\n", (void*)s->ssm_conv_state);
    else
        printf("  ssm_conv_state:   NULL\n");
    if (s->ssm_hidden_state)
        printf("  ssm_hidden_state: ptr=%p\n", (void*)s->ssm_hidden_state);
    else
        printf("  ssm_hidden_state: NULL\n");

    /* ── Run forward pass for n_gen tokens ── */
    printf("\n=== Forward Pass (%d tokens) ===\n", n_gen);
    int start_token = 1; /* BOS */
    /* Check token_embd table health */
    float test_vec[32];
    embed_row(test_vec, s->w.token_embd, s->w.t_embd, 0, 32, cfg->n_vocab);
    float t_min = INFINITY, t_max = -INFINITY; int t_nan = 0;
    for (int i = 0; i < 32; i++) {
        if (isnan(test_vec[i])) t_nan++;
        if (test_vec[i] < t_min) t_min = test_vec[i];
        if (test_vec[i] > t_max) t_max = test_vec[i];
    }
    printf("  token_embd[0][0..31]: nan=%d range=[%e, %e]\n", t_nan, t_min, t_max);
    embed_row(test_vec, s->w.token_embd, s->w.t_embd, 1, 32, cfg->n_vocab);
    t_min = INFINITY; t_max = -INFINITY; t_nan = 0;
    for (int i = 0; i < 32; i++) {
        if (isnan(test_vec[i])) t_nan++;
        if (test_vec[i] < t_min) t_min = test_vec[i];
        if (test_vec[i] > t_max) t_max = test_vec[i];
    }
    printf("  token_embd[1][0..31]: nan=%d range=[%e, %e]\n", t_nan, t_min, t_max);
    const int E = cfg->n_embd;
    float* hidden = (float*)malloc((size_t)E * sizeof(float));
    float* output = (float*)malloc((size_t)E * sizeof(float));
    if (!hidden || !output) {
        fprintf(stderr, "OOM\n");
        free(hidden); free(output);
        ct_infer_free(s);
        ct_gguf_close(gguf);
        return 1;
    }

    for (int pos = 0; pos < n_gen; pos++) {
        /* Embed current token */
        embed_row(hidden, s->w.token_embd, s->w.t_embd,
                  pos == 0 ? start_token : 0, E, cfg->n_vocab);

        /* Forward pass */
        clock_t t0 = clock();
        int ret = ct_infer_forward(s, pos, hidden, output);
        clock_t t1 = clock();
        double ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

        if (ret != 0) {
            fprintf(stderr, "FAIL: ct_infer_forward returned %d at pos %d\n", ret, pos);
            break;
        }

        /* Run final norm + output projection to get logits */
        rms_norm(s->normed, output, s->w.final_norm, E, cfg->norm_rms_eps);
        matmul(s->logits, s->normed, s->w.output_weight ? s->w.output_weight : s->w.token_embd,
               s->w.output_weight ? s->w.t_out : s->w.t_embd,
               E, cfg->n_vocab);

        printf("  token %3d: %.2f ms", pos, ms);
        print_logit_stats(s->logits, cfg->n_vocab, "logits");

            /* Debug: print hidden state stats after forward pass */
        {
            float h_emin = INFINITY, h_emax = -INFINITY;
            int h_enan = 0, h_einf = 0;
            for (int i = 0; i < E; i++) {
                if (isnan(output[i])) h_enan++;
                if (isinf(output[i])) h_einf++;
                if (output[i] < h_emin) h_emin = output[i];
                if (output[i] > h_emax) h_emax = output[i];
            }
            printf("  hidden_out: nan=%d inf=%d [%e, %e]\n",
                   h_enan, h_einf, h_emin, h_emax);
        }

        /* Use next hidden as input for next step (copy) */
        memcpy(hidden, output, (size_t)E * sizeof(float));
    }

    /* ── Cleanup ── */
    free(hidden);
    free(output);
    ct_infer_free(s);
    ct_gguf_close(gguf);

    printf("\n=== PASS ===\n");
    return 0;
}

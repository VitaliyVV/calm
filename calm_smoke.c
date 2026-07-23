/* Minimal smoke test: load model, run 1 token, check no NaN/Inf */
#include "calm_infer.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: calm_smoke <model.gguf>\n"); return 1; }
    
    ct_gguf_context* gguf = ct_gguf_open(argv[1]);
    if (!gguf) { fprintf(stderr, "FAIL: ct_gguf_open\n"); return 1; }
    
    ct_infer_state* s = ct_infer_create(gguf, 128, 0);
    if (!s) { fprintf(stderr, "FAIL: ct_infer_create\n"); return 1; }
    
    ct_infer_config* cfg = &s->w.config;
    printf("Model: layers=%d embd=%d head=%d hkv=%d hd=%d ff=%d vocab=%d\n",
           cfg->n_layer, cfg->n_embd, cfg->n_head, cfg->n_head_kv,
           cfg->head_dim, cfg->n_ff, cfg->n_vocab);
    
    /* Test 1: single token (BOS) */
    int bos = 128000; /* Llama BOS */
    printf("\n--- Test 1: Single token (BOS=%d) ---\n", bos);
    embed_row(s->hidden, s->w.token_embd, s->w.t_embd, bos, cfg->n_embd);
    
    /* Verify embedding is sane */
    float sum = 0, max_abs = 0, nan_count = 0;
    for (int i = 0; i < cfg->n_embd; i++) {
        sum += s->hidden[i];
        if (fabsf(s->hidden[i]) > max_abs) max_abs = fabsf(s->hidden[i]);
        if (isnan(s->hidden[i]) || isinf(s->hidden[i])) nan_count++;
    }
    printf("  emb: sum=%.2f max_abs=%.2f nan=%0.f first4=[%.4f,%.4f,%.4f,%.4f]\n",
           sum, max_abs, nan_count, s->hidden[0], s->hidden[1], s->hidden[2], s->hidden[3]);
    
    /* Run 1-layer forward pass */
    float* out = s->buf_q;
    if (ct_infer_forward(s, 0, s->hidden, out) != 0) {
        fprintf(stderr, "FAIL: ct_infer_forward\n"); return 1;
    }
    
    /* Check output sanity */
    float o_sum = 0, o_max_abs = 0, o_nan = 0;
    for (int i = 0; i < cfg->n_embd; i++) {
        o_sum += out[i];
        if (fabsf(out[i]) > o_max_abs) o_max_abs = fabsf(out[i]);
        if (isnan(out[i]) || isinf(out[i])) o_nan++;
    }
    printf("  L0 out: sum=%.2f max_abs=%.2f nan=%0.f first4=[%.4f,%.4f,%.4f,%.4f]\n",
           o_sum, o_max_abs, o_nan, out[0], out[1], out[2], out[3]);
    
    if (o_nan > 0 || o_max_abs > 1e6) {
        printf("FAIL: numerical instability\n");
        return 1;
    }
    
    /* Test 2: special token (<|start_header_id|> for Llama) */
    int specl = 128006;
    printf("\n--- Test 2: Special token (%d) ---\n", specl);
    embed_row(s->hidden, s->w.token_embd, s->w.t_embd, specl, cfg->n_embd);
    sum = 0; max_abs = 0; nan_count = 0;
    for (int i = 0; i < cfg->n_embd; i++) {
        sum += s->hidden[i];
        if (fabsf(s->hidden[i]) > max_abs) max_abs = fabsf(s->hidden[i]);
        if (isnan(s->hidden[i]) || isinf(s->hidden[i])) nan_count++;
    }
    printf("  emb: sum=%.2f max_abs=%.2f nan=%0.f first4=[%.4f,%.4f,%.4f,%.4f]\n",
           sum, max_abs, nan_count, s->hidden[0], s->hidden[1], s->hidden[2], s->hidden[3]);
    
    if (nan_count > 0 || max_abs > 1e6) {
        printf("FAIL: embedding has NaN or extreme values\n");
        return 1;
    }
    
    printf("\nPASS\n");
    ct_infer_free(s);
    ct_gguf_close(gguf);
    return 0;
}

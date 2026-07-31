/**
 * calm_mla_test.c — Synthetic equivalence test for ct_forward_mla.
 *
 * Builds a tiny MLA model (mini DeepSeek2-Lite dims, dc = 64 so the Q8_0
 * quantized Wkv_b spans 2×32-blocks per column — exercises the block
 * addressing that was broken for dc > 256 and for 32-element types).
 *
 * Checks, per forward (positions 0..3, fresh random inputs):
 *   1. ct_forward_mla (F32 Wkv_b)  == independent F32 reference  (tol ~1e-3)
 *   2. ct_forward_mla (Q8_0 Wkv_b) == independent F32 reference  (tol ~5%)
 * across three norm combinations: k_norm OFF/ON, kv_a_norm OFF/ON.
 *
 * The reference expands K/V per position directly from the GGUF F32 layout
 * (w[c*dc + k]) — it shares primitives (matmul/rope/rms_norm) with the
 * implementation but not its orchestration, so it independently validates
 * cache layout, block addressing, k_norm semantics and softmax/V paths.
 *
 * Build (MSVC):  cl /nologo /O1 /std:c11 /arch:AVX2 /DCT_AVX2 /W4 /Fe:calm_mla_test.exe
 *                calm_mla_test.c calm_mla.c calm_infer.c calm_gguf.c calm_quant.c
 *                calm_ssm.c calm_tokenizer.c calm_tools.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "calm_infer.h"
#include "calm_mla.h"
#include "calm_quant.h"
#include "calm_gguf.h"

/* ── primitives from calm_infer.c (shared by impl and reference) ── */
extern void matmul(float* y, const float* x, const void* w, int type, int I, int O);
extern void rope(float* buf, int n, int pos, float base);
extern void rms_norm(float* y, const float* x, const float* w, int n, float eps);

/* ── tiny deterministic RNG ── */
static uint32_t g_rng = 0x9E3779B9u;
static float frand(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xFFFF) / 32768.0f) - 1.0f;
}

/* ── mini DeepSeek2-Lite dims ── */
#define E  64
#define H  4
#define HK 2
#define DC 64          /* kv_lora_rank — 64 = 2 Q8_0 blocks per column */
#define DR 16          /* qk_rope_head_dim */
#define DN 32          /* qk_nope_head_dim */
#define DV 32          /* value_head_dim */
#define QL 16          /* q_lora_rank (decomposed Q would use this; fused here) */
#define CTX 8          /* max_ctx */
#define HD (DN + DR)
#define TPH (DN + DV)  /* total_per_head */
#define TC (H * TPH)   /* total_cols */

static ct_infer_config g_cfg;
static ct_infer_layer g_lw_f32, g_lw_q8;
static float g_qw[E * H * HD];                    /* fused Q  [E, H*hd] */
static float g_kva_w[E * (DC + DR)];              /* KV latent [E, dc+dr] */
static float g_kvb_f32[DC * TC];                  /* Wkv_b F32 [dc, total_cols] GGUF layout */
static char g_q8_buf[TC * (DC / 32) * sizeof(ct_block_q8_0)];
static float g_kva_norm_w[DC + DR];
static float g_k_norm_w[DN];

/* ── Independent F32 reference ── */
static void reference_mla(float* out, const float* normed, const ct_infer_layer* lw,
                          const ct_infer_config* cfg, float* cache, int pos, int max_ctx) {
    const int hd = HD, dc = DC, dr = DR, dn = DN, dv = DV;
    const int total_per_head = TPH;
    const float* w = (const float*)lw->attn_kv_b; /* GGUF: element (k,c) at w[c*dc + k] */
    float rcp = 1.0f / sqrtf((float)hd);
    int n_groups = H / HK;
    if (n_groups < 1) n_groups = 1;

    float qb[512], kva[160], kr[64], kk[DN], scores[512];

    /* Q (fused path in this test) + rope on the rope part */
    matmul(qb, normed, lw->attn_q, CT_GGUF_TYPE_F32, E, H * hd);
    for (int hh = 0; hh < H; hh++)
        rope(qb + hh * hd + dn, dr, pos, cfg->rope_freq_base);

    /* KV latent + full-latent norm */
    matmul(kva, normed, lw->attn_kv_a, CT_GGUF_TYPE_F32, E, dc + dr);
    if (lw->attn_kv_a_norm)
        rms_norm(kva, kva, lw->attn_kv_a_norm, dc + dr, cfg->norm_rms_eps);

    /* Store [c | k_rope] in cache */
    for (int d = 0; d < dc; d++)
        cache[(size_t)d * max_ctx + pos] = kva[d];
    memcpy(kr, kva + dc, (size_t)dr * sizeof(float));
    rope(kr, dr, pos, cfg->rope_freq_base);
    for (int d = 0; d < dr; d++)
        cache[(size_t)(dc + d) * max_ctx + pos] = kr[d];

    /* K_nope norm: per-head-group RMS of the K expansion of the current latent
     * (plain RMS, weight not applied — matches the implementation convention). */
    if (lw->attn_k_norm) {
        for (int kg = 0; kg < HK; kg++) {
            for (int d = 0; d < dn; d++) {
                const float* col = w + (size_t)(kg * total_per_head + d) * dc;
                float s = 0.0f;
                for (int k = 0; k < dc; k++) s += kva[k] * col[k];
                kk[d] = s;
            }
            float ss = 0.0f;
            for (int d = 0; d < dn; d++) ss += kk[d] * kk[d];
            cache[(size_t)(dc + dr + kg) * max_ctx + pos] = sqrtf(ss / (float)dn + 1e-6f);
        }
    }

    /* Per-head attention */
    for (int hh = 0; hh < H; hh++) {
        int kg = hh / n_groups;
        const float* Qh = qb + hh * hd;
        const float* Qr = Qh + dn;
        float mx = -1e38f;

        for (int p = 0; p <= pos; p++) {
            const float* cp = cache + (size_t)p;
            float dot_nope = 0.0f;
            for (int d = 0; d < dn; d++) {
                const float* col = w + (size_t)(kg * total_per_head + d) * dc;
                float kd = 0.0f;
                for (int k = 0; k < dc; k++) kd += cp[(size_t)k * max_ctx] * col[k];
                dot_nope += Qh[d] * kd;
            }
            if (lw->attn_k_norm)
                dot_nope /= cache[(size_t)(dc + dr + kg) * max_ctx + p];
            float dot_rope = 0.0f;
            const float* krp = cache + (size_t)dc * max_ctx + p;
            for (int d = 0; d < dr; d++)
                dot_rope += Qr[d] * krp[(size_t)d * max_ctx];
            scores[p] = (dot_nope + dot_rope) * rcp;
            if (scores[p] > mx) mx = scores[p];
        }

        float se = 0.0f;
        for (int p = 0; p <= pos; p++) { scores[p] = expf(scores[p] - mx); se += scores[p]; }
        float rs = 1.0f / (se + 1e-10f);
        for (int p = 0; p <= pos; p++) scores[p] *= rs;

        float* outh = out + hh * dv;
        for (int d = 0; d < dv; d++) {
            const float* col = w + (size_t)(kg * total_per_head + dn + d) * dc;
            float acc = 0.0f;
            for (int p = 0; p <= pos; p++) {
                float sp = scores[p];
                const float* cp = cache + (size_t)p;
                float cw = 0.0f;
                for (int k = 0; k < dc; k++) cw += cp[(size_t)k * max_ctx] * col[k];
                acc += sp * cw;
            }
            outh[d] = acc;
        }
    }
}

static void init_weights(void) {
    for (int i = 0; i < E * H * HD; i++)            g_qw[i] = frand();
    for (int i = 0; i < E * (DC + DR); i++)         g_kva_w[i] = frand();
    for (int i = 0; i < DC * TC; i++)               g_kvb_f32[i] = frand();
    for (int i = 0; i < DC + DR; i++)               g_kva_norm_w[i] = 1.0f;
    for (int i = 0; i < DN; i++)                    g_k_norm_w[i] = 1.0f;

    /* Quantize Wkv_b columns to Q8_0 in GGUF layout: block index c*nb + kb */
    int nb = DC / 32;
    ct_block_q8_0* q = (ct_block_q8_0*)g_q8_buf;
    for (int c = 0; c < TC; c++)
        for (int kb = 0; kb < nb; kb++)
            ct_quant_q8_0(g_kvb_f32 + (size_t)c * DC + kb * 32, q + (size_t)c * nb + kb, 32);

    memset(&g_lw_f32, 0, sizeof(g_lw_f32));
    g_lw_f32.attn_q = g_qw;     g_lw_f32.t_q     = CT_GGUF_TYPE_F32;
    g_lw_f32.attn_kv_a = g_kva_w; g_lw_f32.t_kva = CT_GGUF_TYPE_F32;
    g_lw_f32.attn_kv_b = g_kvb_f32; g_lw_f32.t_kvb = CT_GGUF_TYPE_F32;
    g_lw_q8 = g_lw_f32;
    g_lw_q8.attn_kv_b = g_q8_buf; g_lw_q8.t_kvb = CT_GGUF_TYPE_Q8_0;

    memset(&g_cfg, 0, sizeof(g_cfg));
    g_cfg.n_embd = E;
    g_cfg.n_head = H;
    g_cfg.n_head_kv = HK;
    g_cfg.mla_kv_lora_rank = DC;
    g_cfg.mla_q_lora_rank = QL;
    g_cfg.mla_qk_nope_head_dim = DN;
    g_cfg.mla_qk_rope_head_dim = DR;
    g_cfg.mla_v_head_dim = DV;
    g_cfg.rope_freq_base = 10000.0f;
    g_cfg.norm_rms_eps = 1e-6f;
}

typedef struct { const char* name; int k_norm; int kv_a_norm; } combo_t;

/* Diagnostic: verify the Q8_0 quantization used by the test round-trips
 * through ct_dequant_q8_0 with the same (c*nb + kb) block indexing. */
static void verify_q8_roundtrip(void) {
    const int nb = DC / 32;
    const ct_block_q8_0* q = (const ct_block_q8_0*)g_q8_buf;
    float max_err = 0.0f, max_val = 0.0f;
    for (int c = 0; c < TC; c++) {
        for (int kb = 0; kb < nb; kb++) {
            float back[32];
            ct_dequant_q8_0(&q[(size_t)c * nb + kb], back, 32);
            for (int i = 0; i < 32; i++) {
                float e = fabsf(back[i] - g_kvb_f32[(size_t)c * DC + kb * 32 + i]);
                float v = fabsf(g_kvb_f32[(size_t)c * DC + kb * 32 + i]);
                if (e > max_err) max_err = e;
                if (v > max_val) max_val = v;
            }
        }
    }
    printf("Q8_0 roundtrip: max_err=%.3e (max|w|=%.2f)\n", max_err, max_val);
}

/* Diagnostic: print first 4 outputs of each path for one forward. */
static int run_combo(const combo_t* cmb) {
    const int rows = DC + DR + HK;
    float* cache_ref = (float*)calloc((size_t)rows * CTX, sizeof(float));
    float* cache_f32 = (float*)calloc((size_t)rows * CTX, sizeof(float));
    float* cache_q8  = (float*)calloc((size_t)rows * CTX, sizeof(float));
    if (!cache_ref || !cache_f32 || !cache_q8) { printf("OOM\n"); return 1; }

    g_lw_f32.attn_k_norm = cmb->k_norm ? g_k_norm_w : NULL;
    g_lw_f32.attn_kv_a_norm = cmb->kv_a_norm ? g_kva_norm_w : NULL;
    g_lw_q8.attn_k_norm = g_lw_f32.attn_k_norm;
    g_lw_q8.attn_kv_a_norm = g_lw_f32.attn_kv_a_norm;

    double max_e_ref = 0.0, max_e_q8 = 0.0, ref_scale = 0.0;
    int fail = 0;

    for (int pos = 0; pos < 4; pos++) {
        float x[E];
        for (int i = 0; i < E; i++) x[i] = frand();

        float out_ref[H * HD], out_f32[H * HD], out_q8[H * HD];
        /* NB: buf_q receives H*(dn+dr) = H*HD floats (Q projection) even though
         * the attention output occupies only the first H*DV of them. */
        reference_mla(out_ref, x, &g_lw_f32, &g_cfg, cache_ref, pos, CTX);
        ct_forward_mla(out_f32, x, &g_lw_f32, &g_cfg, cache_f32, pos, CTX);
        ct_forward_mla(out_q8,  x, &g_lw_q8,  &g_cfg, cache_q8,  pos, CTX);

        for (int i = 0; i < H * DV; i++) {
            double d_ref = fabs((double)out_f32[i] - out_ref[i]);
            double d_q8  = fabs((double)out_q8[i]  - out_ref[i]);
            double ar    = fabs((double)out_ref[i]);
            if (d_ref > max_e_ref) max_e_ref = d_ref;
            if (d_q8  > max_e_q8)  max_e_q8  = d_q8;
            if (ar    > ref_scale) ref_scale = ar;
            if (d_ref > 1e-3) fail |= 1;
            if (d_q8 > 0.05 * ref_scale + 1e-3) fail |= 2;
        }
    }

    printf("[%s] F32 vs ref: max_abs=%.3e  Q8_0 vs ref: max_abs=%.3e (ref scale %.2f)  %s\n",
           cmb->name, max_e_ref, max_e_q8, ref_scale, fail ? "FAIL" : "PASS");

    free(cache_ref); free(cache_f32); free(cache_q8);
    return fail != 0;
}

int main(void) {
    static const combo_t combos[] = {
        { "k_norm=OFF kv_a_norm=OFF", 0, 0 },
        { "k_norm=ON  kv_a_norm=OFF", 1, 0 },
        { "k_norm=ON  kv_a_norm=ON",  1, 1 },
    };
    ct_quant_init();   /* fp16 scale table — required by quant/dequant */
    init_weights();
    verify_q8_roundtrip();
    int fails = 0;
    for (size_t i = 0; i < sizeof(combos) / sizeof(combos[0]); i++) {
        fails += run_combo(&combos[i]);
    }
    printf(fails ? "MLA TEST: %d COMBO(S) FAILED\n" : "MLA TEST: ALL PASS\n", fails);
    return fails ? 1 : 0;
}

#include "calm_gguf.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    
    ct_gguf_context* ctx = ct_gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    
    const char* arch = ct_gguf_architecture(ctx);
    printf("Architecture: %s\n", arch ? arch : "null");
    
    // Helper: build full key from arch + key suffix
    char key[256];
    #define META_A(suffix, def) ({ \
        snprintf(key, sizeof(key), "%s.%s", arch ? arch : "unknown", suffix); \
        (int)ct_gguf_meta_uint(ctx, key, def); })
    
    printf("n_layer: %d\n", META_A("block_count", 0));
    printf("n_embd: %d\n", META_A("embedding_length", 0));
    printf("n_head: %d\n", META_A("attention.head_count", 0));
    printf("n_head_kv: %d\n", META_A("attention.head_count_kv", 0));
    printf("n_ff: %d\n", META_A("feed_forward_length", 0));
    printf("head_dim: %d\n", META_A("attention.head_dim", 0));
    printf("ssm.conv_kernel: %d\n", META_A("ssm.conv_kernel", 0));
    printf("ssm.inner_size: %d\n", META_A("ssm.inner_size", 0));
    printf("ssm.state_size: %d\n", META_A("ssm.state_size", 0));
    printf("ssm.time_step_rank: %d\n", META_A("ssm.time_step_rank", 0));
    printf("ssm.dt_b_c_rms: %d\n", META_A("ssm.dt_b_c_rms", 0));
    
    // Dump raw metadata keys
    printf("\nAll raw metadata:\n");
    for (unsigned i = 0; i < ctx->metadata.count && i < 50; i++) {
        printf("  [%u] %s (type=%d)\n", i, ctx->metadata.keys[i], ctx->metadata.types[i]);
    }
    
    // Tensors
    printf("\nTotal tensors: %llu\n", (unsigned long long)ctx->tensor_count);
    
    printf("\nLayer 0 tensors:\n");
    for (unsigned i = 0; i < ctx->tensor_count; i++) {
        ct_gguf_tensor_info* t = &ctx->tensors[i];
        if (strstr(t->name, "blk.0.")) {
            printf("  %s type=%d dims=%d [", t->name, t->type, t->n_dims);
            for (int d = 0; d < t->n_dims; d++)
                printf("%llu ", (unsigned long long)t->dims[d]);
            printf("]\n");
        }
    }
    
    // Check specific tensors by name
    printf("\nKey tensor search:\n");
    const char* checks[] = {
        "blk.0.ssm_qkv.weight", "blk.0.ssm_conv1d.weight", "blk.0.ssm_alpha.weight",
        "blk.0.ssm_beta.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
        "blk.0.ssm_out.weight", "blk.0.ssm_conv1d.bias", "blk.0.attn_q.weight",
        NULL
    };
    for (int c = 0; checks[c]; c++) {
        const ct_gguf_tensor_info* t = ct_gguf_find_tensor(ctx, checks[c]);
        if (t) {
            printf("  ✅ %s type=%d dims=%d [", t->name, t->type, t->n_dims);
            for (int d = 0; d < t->n_dims; d++) 
                printf("%llu ", (unsigned long long)t->dims[d]);
            printf("]\n");
        } else {
            printf("  ❌ %s NOT FOUND\n", checks[c]);
        }
    }
    
    ct_gguf_close(ctx);
    return 0;
}

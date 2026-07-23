/**
 * dump_tensors.c — Dump GGUF tensor names and metadata
 * Compile: clang -O0 -g -std=c11 -o dump_tensors dump_tensors.c calm_gguf.c calm_quant.c -lm
 */
#include "calm_gguf.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    ct_gguf_context* ctx = ct_gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "FAIL: open\n"); return 1; }

    printf("Architecture: %s\n\n", ct_gguf_architecture(ctx));

    /* Direct struct access to iterate tensors */
    printf("Tensor count: %llu\n\n", (unsigned long long)ctx->tensor_count);

    /* Print key metadata */
    printf("=== Metadata ===\n");
    for (int i = 0; i < ctx->metadata.count; i++) {
        printf("  %s", ctx->metadata.keys[i]);
        switch (ctx->metadata.types[i]) {
            case CT_GGUF_VALUE_UINT8:  printf(" = uint8(%u)\n", ctx->metadata.values[i].v_u8); break;
            case CT_GGUF_VALUE_INT8:   printf(" = int8(%d)\n", ctx->metadata.values[i].v_i8); break;
            case CT_GGUF_VALUE_UINT32: printf(" = uint32(%u)\n", ctx->metadata.values[i].v_u32); break;
            case CT_GGUF_VALUE_INT32:  printf(" = int32(%d)\n", ctx->metadata.values[i].v_i32); break;
            case CT_GGUF_VALUE_FLOAT32: printf(" = float32(%f)\n", ctx->metadata.values[i].v_f32); break;
            case CT_GGUF_VALUE_UINT64: printf(" = uint64(%llu)\n", (unsigned long long)ctx->metadata.values[i].v_u64); break;
            case CT_GGUF_VALUE_INT64:  printf(" = int64(%lld)\n", (long long)ctx->metadata.values[i].v_i64); break;
            case CT_GGUF_VALUE_FLOAT64: printf(" = float64(%f)\n", ctx->metadata.values[i].v_f64); break;
            case CT_GGUF_VALUE_BOOL:   printf(" = bool(%s)\n", ctx->metadata.values[i].v_bool?"true":"false"); break;
            case CT_GGUF_VALUE_STRING: printf(" = \"%s\"\n", ctx->metadata.values[i].v_str ? ctx->metadata.values[i].v_str : "NULL"); break;
            case CT_GGUF_VALUE_ARRAY:  printf(" = array(...)\n"); break;
            default: printf(" = type=%d\n", ctx->metadata.types[i]); break;
        }
    }

    printf("\n=== Tensors ===\n");
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        const ct_gguf_tensor_info* t = &ctx->tensors[i];
        /* Look for expert-related tensors */
        printf("  [%3llu] type=%2d %s [", (unsigned long long)i, t->type, t->name);
        for (int d = 0; d < t->n_dims; d++)
            printf("%llu%s", (unsigned long long)t->dims[d], d+1<t->n_dims ? "," : "");
        printf("]\n");
    }

    ct_gguf_close(ctx);
    return 0;
}

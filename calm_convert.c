/**
 * calm_convert.c — Calm Model Converter
 *
 * Преобразует существующие GGUF модели в BQ1_0 (binary 1-bit)
 * или TQ1_0 (ternary 1.58-bit) формат для запуска на устройствах
 * с ограниченной памятью (3-4 GB RAM).
 *
 * Использование:
 *   calm_convert --input model.gguf --format bq1_0 --output model-bq1_0.gguf
 *   calm_convert --input model.gguf --format tq1_0 --output model-tq1_0.gguf
 *
 * Сборка:
 *   clang -O3 -std=c11 -march=native -o calm_convert calm_convert.c calm_gguf.c calm_quant.c -lm
 */
#define _GNU_SOURCE
#include "calm_gguf.h"
#include "calm_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

/* ═══════════════════════════════════════════════════════════════
 * Tensor classification: which tensors to quantize to 1-bit
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    CT_TENSOR_SKIP,      /* Keep as-is (tiny, norms, etc.) */
    CT_TENSOR_WEIGHT,    /* Quantize to 1-bit/ternary */
    CT_TENSOR_HEAD,      /* Quantize to Q8_0 (lm_head, embed) */
} ct_tensor_class;

/* Sensitive tensors that should stay higher precision */
static const char* ct_skip_tensors[] = {
    "token_embd.weight",
    "output.weight",
    "tok_embeddings.weight",
    "embed.weight",
    "blk.0.attn_norm.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.attn_norm.bias",
    "blk.0.ffn_norm.bias",
    "blk.0.norm.weight",
    "blk.0.norm.bias",
    NULL
};

/* Check if tensor name matches a skip pattern */
static bool should_skip(const char* name) {
    if (!name) return false;
    /* Skip norms/biases (tiny tensors) */
    if (strstr(name, "norm") || strstr(name, "_norm") ||
        strstr(name, ".bias") || strstr(name, "rope") ||
        strstr(name, "freqs")) {
        /* But only if it's small (< 100K elements) */
        return true;  /* We'll check size later */
    }
    return false;
}

/* Check if tensor name should be high precision (embed/head) */
static bool is_head_tensor(const char* name) {
    if (!name) return false;
    if (strstr(name, "token_embd") || strstr(name, "output") ||
        strstr(name, "tok_embed") || strstr(name, "embed") ||
        strstr(name, "head")) {
        return true;
    }
    return false;
}

/* ═══════════════════════════════════════════════════════════════
 * Dequantize a tensor row to float
 * ═══════════════════════════════════════════════════════════════ */

static void dequantize_row(const void* data, int src_type,
                            float* out, int cols) {
    switch (src_type) {
        case CT_GGUF_TYPE_F32: {
            memcpy(out, data, (size_t)cols * 4);
            break;
        }
        case CT_GGUF_TYPE_F16: {
            const uint16_t* f16 = (const uint16_t*)data;
            for (int i = 0; i < cols; i++)
                out[i] = ct_fp16_to_fp32(f16[i]);
            break;
        }
        case CT_GGUF_TYPE_Q8_0: {
            int blk = (cols + 31) / 32;
            for (int b = 0; b < blk; b++) {
                const ct_block_q8_0* block = &((const ct_block_q8_0*)data)[b];
                float d = ct_fp16_to_fp32(block->d);
                for (int j = 0; j < 32 && b * 32 + j < cols; j++)
                    out[b * 32 + j] = block->qs[j] * d;
            }
            break;
        }
        case CT_GGUF_TYPE_Q4_0: {
            int blk = (cols + 31) / 32;
            for (int b = 0; b < blk; b++) {
                const ct_block_q4_0* block = &((const ct_block_q4_0*)data)[b];
                float d = ct_fp16_to_fp32(block->d);
                for (int j = 0; j < 32 && b * 32 + j < cols; j++) {
                    int nib = (block->qs[j >> 1] >> ((j & 1) << 2)) & 0xF;
                    out[b * 32 + j] = ((float)nib - 8.0f) * d;
                }
            }
            break;
        }
        case CT_GGUF_TYPE_Q4_K: {
            /* K-quant 4-bit: simplified dequant (256-block) */
            /* For conversion, we approximate — exact K-quant dequant
             * would need the full llama.cpp reference.
             * Use a block-based approximation. */
            /* Q4_K block: struct { uint16_t d, uint16_t dmin, uint8_t scales[12], uint8_t qs[128] } */
            int blk64 = (cols + 255) / 256;
            for (int b = 0; b < blk64; b++) {
                const uint8_t* bp = (const uint8_t*)data + (size_t)b * 144;
                uint16_t d_raw, dmin_raw;
                memcpy(&d_raw, bp, 2);
                memcpy(&dmin_raw, bp + 2, 2);
                float d = ct_fp16_to_fp32(d_raw);
                float dmin = ct_fp16_to_fp32(dmin_raw);
                const uint8_t* scales = bp + 4;
                const uint8_t* qs = bp + 16;

                for (int j = 0; j < 256 && b * 256 + j < cols; j++) {
                    int sub = j / 32;
                    int subpos = j % 32;
                    /* Decode 6-bit scale: 3 bytes per 4 sub-blocks */
                    int sc_idx = sub / 4;
                    int sc_shift = (sub % 4) * 6;  /* 6 bits per sub-block scale */
                    int sc_raw = (scales[sc_idx * 3] >> (sc_shift & 7)) |
                                 (scales[sc_idx * 3 + 1] << (8 - (sc_shift & 7)));
                    if (sc_shift > 16)
                        sc_raw = (scales[sc_idx * 3 + 1] >> ((sc_shift - 8) & 7)) |
                                  (scales[sc_idx * 3 + 2] << (8 - ((sc_shift - 8) & 7)));
                    sc_raw &= 0x3F;
                    float sc_d = d * (float)(sc_raw - 16);
                    float sc_m = dmin * (float)(sc_raw - 16);

                    int byte_idx = j / 2;
                    int nib_shift = (j & 1) ? 4 : 0;
                    int qv = (qs[byte_idx] >> nib_shift) & 0xF;
                    out[b * 256 + j] = qv * sc_d + sc_m;
                }
            }
            break;
        }
        default: {
            /* Unknown format: zero out */
            for (int i = 0; i < cols; i++)
                out[i] = 0;
            fprintf(stderr, "  Warning: unsupported source type %d, zeroing\n", src_type);
            break;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Convert a single tensor
 * ═══════════════════════════════════════════════════════════════ */

static size_t convert_tensor(const ct_gguf_tensor_info* src_tensor,
                              const void* src_data,
                              int dst_type,
                              void* dst_buf,
                              ct_tensor_class tclass) {
    int rows = (int)src_tensor->dims[1];
    int cols = (int)src_tensor->dims[0];

    if (tclass == CT_TENSOR_SKIP) {
        /* Keep original data */
        memcpy(dst_buf, src_data, src_tensor->size);
        return src_tensor->size;
    }

    if (tclass == CT_TENSOR_HEAD && dst_type != CT_GGUF_TYPE_BQ1_0) {
        /* Head tensors: keep original or convert to Q8_0 */
        if (src_tensor->type == CT_GGUF_TYPE_Q8_0 ||
            src_tensor->type == CT_GGUF_TYPE_F32 ||
            src_tensor->type == CT_GGUF_TYPE_F16) {
            memcpy(dst_buf, src_data, src_tensor->size);
            return src_tensor->size;
        }
        /* Convert to Q8_0 */
        float* row_buf = malloc((size_t)cols * 4);
        size_t dst_size = 0;
        for (int r = 0; r < rows; r++) {
            dequantize_row((const uint8_t*)src_data + (size_t)r * src_tensor->size / rows,
                           src_tensor->type, row_buf, cols);
             int nblk = (cols + 31) / 32;
            ct_block_q8_0* blk = (ct_block_q8_0*)((uint8_t*)dst_buf + dst_size);
            for (int b = 0; b < nblk; b++) {
                ct_quant_q8_0(row_buf + b * 32, &blk[b],
                              (b + 1) * 32 <= cols ? 32 : cols - b * 32);
            }
            dst_size += (size_t)nblk * CT_SIZEOF_Q8_0;
        }
        free(row_buf);
        return dst_size;
    }

    /* --- Main conversion path: dequantize → requantize to BQ1_0/TQ1_0 --- */
    float* row_buf = malloc((size_t)cols * 4);
    size_t dst_size = 0;

    for (int r = 0; r < rows; r++) {
        const uint8_t* src_row = (const uint8_t*)src_data +
            (size_t)r * src_tensor->size / rows;
        dequantize_row(src_row, src_tensor->type, row_buf, cols);

        if (dst_type == CT_GGUF_TYPE_BQ1_0) {
            int ng = (cols + 127) / 128;
            ct_block_bq1_0* blk = (ct_block_bq1_0*)((uint8_t*)dst_buf + dst_size);
            for (int g = 0; g < ng; g++) {
                int count = (g + 1) * 128 <= cols ? 128 : cols - g * 128;
                ct_quant_bq1_0(row_buf + g * 128, &blk[g], count);
            }
            dst_size += (size_t)ng * sizeof(ct_block_bq1_0);
        } else if (dst_type == CT_GGUF_TYPE_TQ1_0) {
            int nb = (cols + 255) / 256;
            ct_block_tq1_0* blk = (ct_block_tq1_0*)((uint8_t*)dst_buf + dst_size);
            for (int b = 0; b < nb; b++) {
                int count = (b + 1) * 256 <= cols ? 256 : cols - b * 256;
                ct_quant_tq1_0(row_buf + b * 256, &blk[b], count);
            }
            dst_size += (size_t)nb * sizeof(ct_block_tq1_0);
        } else {
            memcpy(dst_buf, src_data, src_tensor->size);
            dst_size = src_tensor->size;
        }
    }

    free(row_buf);
    return dst_size;
}

/* ═══════════════════════════════════════════════════════════════
 * GGUF Writer
 * ═══════════════════════════════════════════════════════════════ */

static bool write_gguf(const char* path,
                        const ct_gguf_context* src,
                        const ct_gguf_tensor_info** tensors,
                        const void** tensor_data,
                        size_t* tensor_sizes,
                        int* tensor_types,
                        int n_tensors) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot create: %s\n", path);
        return false;
    }

    /* 1. Header */
    uint32_t magic = CT_GGUF_MAGIC;
    uint32_t version = 3;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);

    uint64_t n_tensors_u64 = (uint64_t)n_tensors;
    uint64_t n_metadata = src->metadata_count;
    fwrite(&n_tensors_u64, 8, 1, f);
    fwrite(&n_metadata, 8, 1, f);

    /* 2. Metadata (passthrough from source) */
    /* We need to re-scan the source file for raw metadata bytes.
     * For simplicity, just write the decoded metadata back. */
    /* Find where metadata starts in source */
    size_t meta_start = 4 + 4 + 16; /* magic + version + tensor/meta counts */
    /* For v3: metadata follows header directly. We read it already. */
    /* Re-encode metadata from our parsed representation. */
    char file_buf[65536];
    for (uint64_t i = 0; i < src->metadata_count; i++) {
        /* Key length + key */
        size_t klen = strlen(src->metadata.keys[i]);
        uint64_t klen64 = (uint64_t)klen;
        fwrite(&klen64, 8, 1, f);  /* Write as uint64 (v3) */
        fwrite(src->metadata.keys[i], 1, klen, f);

        /* Value type */
        uint32_t vtype = src->metadata.types[i];
        fwrite(&vtype, 4, 1, f);

        /* Value */
        switch (vtype) {
            case CT_GGUF_VALUE_UINT8: {
                uint8_t v = src->metadata.values[i].v_u8;
                fwrite(&v, 1, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT8: {
                int8_t v = src->metadata.values[i].v_i8;
                fwrite(&v, 1, 1, f);
                break;
            }
            case CT_GGUF_VALUE_UINT16: {
                uint16_t v = src->metadata.values[i].v_u16;
                fwrite(&v, 2, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT16: {
                int16_t v = src->metadata.values[i].v_i16;
                fwrite(&v, 2, 1, f);
                break;
            }
            case CT_GGUF_VALUE_UINT32: {
                uint32_t v = src->metadata.values[i].v_u32;
                fwrite(&v, 4, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT32: {
                int32_t v = src->metadata.values[i].v_i32;
                fwrite(&v, 4, 1, f);
                break;
            }
            case CT_GGUF_VALUE_FLOAT32: {
                float v = src->metadata.values[i].v_f32;
                fwrite(&v, 4, 1, f);
                break;
            }
            case CT_GGUF_VALUE_BOOL: {
                uint8_t v = src->metadata.values[i].v_bool ? 1 : 0;
                fwrite(&v, 1, 1, f);
                break;
            }
            case CT_GGUF_VALUE_STRING: {
                const char* s = src->metadata.values[i].v_str;
                size_t slen = s ? strlen(s) : 0;
                uint64_t slen64 = (uint64_t)slen;
                fwrite(&slen64, 8, 1, f);
                if (slen > 0) fwrite(s, 1, slen, f);
                break;
            }
            case CT_GGUF_VALUE_ARRAY: {
                /* Check if this is a vocabulary array — serialize from vocab struct */
                const char* k = src->metadata.keys[i];
                bool wrote = false;

                if (k && strcmp(k, "tokenizer.ggml.tokens") == 0 && src->vocab.n_vocab > 0) {
                    uint32_t arr_type = 8; /* string */
                    uint64_t arr_len = (uint64_t)src->vocab.n_vocab;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                    for (int vi = 0; vi < src->vocab.n_vocab; vi++) {
                        size_t sl = src->vocab.tokens[vi] ? strlen(src->vocab.tokens[vi]) : 0;
                        uint64_t sl64 = (uint64_t)sl;
                        fwrite(&sl64, 8, 1, f);
                        if (sl > 0) fwrite(src->vocab.tokens[vi], 1, sl, f);
                    }
                    wrote = true;
                }
                if (k && strcmp(k, "tokenizer.ggml.scores") == 0 && src->vocab.n_vocab > 0) {
                    uint32_t arr_type = 6; /* float32 */
                    uint64_t arr_len = (uint64_t)src->vocab.n_vocab;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                    for (int vi = 0; vi < src->vocab.n_vocab; vi++)
                        fwrite(&src->vocab.scores[vi], 4, 1, f);
                    wrote = true;
                }
                if (k && strcmp(k, "tokenizer.ggml.token_type") == 0 && src->vocab.n_vocab > 0) {
                    uint32_t arr_type = 5; /* int32 */
                    uint64_t arr_len = (uint64_t)src->vocab.n_vocab;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                    for (int vi = 0; vi < src->vocab.n_vocab; vi++)
                        fwrite(&src->vocab.token_types[vi], 4, 1, f);
                    wrote = true;
                }

                if (!wrote) {
                    /* Fallback: write empty array */
                    uint32_t arr_type = 0;
                    uint64_t arr_len = 0;
                    fwrite(&arr_type, 4, 1, f);
                    fwrite(&arr_len, 8, 1, f);
                }
                break;
            }
            case CT_GGUF_VALUE_UINT64: {
                uint64_t v = src->metadata.values[i].v_u64;
                fwrite(&v, 8, 1, f);
                break;
            }
            case CT_GGUF_VALUE_INT64: {
                int64_t v = src->metadata.values[i].v_i64;
                fwrite(&v, 8, 1, f);
                break;
            }
            case CT_GGUF_VALUE_FLOAT64: {
                double v = src->metadata.values[i].v_f64;
                fwrite(&v, 8, 1, f);
                break;
            }
            default: break;
        }
    }

    /* 3. Tensor info */
    size_t tensor_data_start = ftell(f);
    /* First pass: write tensor info to calculate offsets */
    size_t current_offset = 0;

    for (int i = 0; i < n_tensors; i++) {
        const ct_gguf_tensor_info* t = tensors[i];
        size_t tname_len = strlen(t->name);
        uint64_t tname_len64 = (uint64_t)tname_len;
        fwrite(&tname_len64, 8, 1, f);
        fwrite(t->name, 1, tname_len, f);

        uint32_t n_dims = t->n_dims;
        int out_type = tensor_types ? tensor_types[i] : (int)t->type;

        /* GGUF v3 order: n_dims → dims → type → offset (NOT n_dims → type → dims!) */
        fwrite(&n_dims, 4, 1, f);

        for (uint32_t d = 0; d < n_dims; d++)
            fwrite(&t->dims[d], 8, 1, f);

        fwrite(&out_type, 4, 1, f);

        /* Write offset (will be recalculated) */
        fwrite(&current_offset, 8, 1, f);
        current_offset += tensor_sizes[i];

        /* Align to 32 bytes */
        while (current_offset % 32 != 0)
            current_offset++;
    }

    /* 4. Tensor data */
    /* Pad to alignment */
    while ((ftell(f) - tensor_data_start) % 32 != 0) {
        uint8_t pad = 0;
        fwrite(&pad, 1, 1, f);
    }

    size_t actual_data_start = ftell(f);
    for (int i = 0; i < n_tensors; i++) {
        /* Write tensor data */
        fwrite(tensor_data[i], 1, tensor_sizes[i], f);
        /* Align to 32 bytes */
        size_t pos = ftell(f);
        size_t data_off = pos - actual_data_start;
        while (data_off % 32 != 0) {
            uint8_t pad = 0;
            fwrite(&pad, 1, 1, f);
            data_off++;
        }
    }

    fclose(f);
    return true;
}

/* ═══════════════════════════════════════════════════════════════
 * Main converter
 * ═══════════════════════════════════════════════════════════════ */

/* Forward decl for win/UNIX compat */
extern int getopt(int argc, char * const argv[], const char *optstring);

int main(int argc, char** argv) {
    const char* input_path = NULL;
    const char* output_path = NULL;
    const char* format = "bq1_0";

    /* Simple arg parsing */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
            format = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Calm Model Converter\n");
            printf("  --input <file.gguf>    Source GGUF model\n");
            printf("  --format <bq1_0|tq1_0> Target format (default: bq1_0)\n");
            printf("  --output <file.gguf>   Output path\n");
            printf("  --help                 This help\n");
            return 0;
        }
    }

    if (!input_path) {
        fprintf(stderr, "Error: --input required\n");
        return 1;
    }
    if (!output_path) {
        fprintf(stderr, "Error: --output required\n");
        return 1;
    }

    int dst_type;
    const char* format_name;
    if (strcmp(format, "bq1_0") == 0) {
        dst_type = CT_GGUF_TYPE_BQ1_0;
        format_name = "BQ1_0 (binary 1-bit)";
    } else if (strcmp(format, "tq1_0") == 0) {
        dst_type = CT_GGUF_TYPE_TQ1_0;
        format_name = "TQ1_0 (ternary 1.58-bit)";
    } else {
        fprintf(stderr, "Error: unknown format '%s'. Use bq1_0 or tq1_0.\n", format);
        return 1;
    }

    /* Init FP16 LUT */
    ct_quant_init();

    printf("Calm Model Converter v0.1\n");
    printf("  Input:  %s\n", input_path);
    printf("  Output: %s\n", output_path);
    printf("  Format: %s\n", format_name);
    printf("\n");

    /* Open source model */
    ct_gguf_context* src = ct_gguf_open(input_path);
    if (!src) {
        fprintf(stderr, "Failed to open: %s\n", input_path);
        return 1;
    }

    printf("  Architecture: %s\n", ct_gguf_architecture(src));
    printf("  Tensors:      %llu\n", (unsigned long long)src->tensor_count);
    printf("  GGUF v%u\n", src->version);
    printf("\n");

    /* Prepare tensor arrays */
    int n = (int)src->tensor_count;
    const ct_gguf_tensor_info** tensors = malloc((size_t)n * sizeof(void*));
    void** data_bufs = malloc((size_t)n * sizeof(void*));
    size_t* sizes = calloc((size_t)n, sizeof(size_t));
    int* out_types = malloc((size_t)n * sizeof(int));
    uint8_t* convert_out = NULL;
    size_t convert_cap = 0;

    /* Estimate worst-case output size */
    size_t total_out = 0;
    for (int i = 0; i < n; i++) {
        tensors[i] = &src->tensors[i];
        const void* tdata = ct_gguf_tensor_data(src, tensors[i]);

        int rows = (int)tensors[i]->dims[1];
        int cols = (int)tensors[i]->dims[0];
        if (tensors[i]->n_dims == 1) {
            rows = 1;
            cols = (int)tensors[i]->dims[0];
        }
        const char* tname = tensors[i]->name;

        /* Classify tensor */
        ct_tensor_class tclass = CT_TENSOR_WEIGHT;
        size_t nelements = (size_t)rows * cols;
        if (should_skip(tname) && nelements < 100000)
            tclass = CT_TENSOR_SKIP;
        else if (is_head_tensor(tname))
            tclass = CT_TENSOR_HEAD;

        /* Determine output type for this tensor */
        if (tclass == CT_TENSOR_SKIP)
            out_types[i] = tensors[i]->type;
        else if (tclass == CT_TENSOR_HEAD)
            out_types[i] = CT_GGUF_TYPE_Q8_0;
        else
            out_types[i] = dst_type;

        /* Allocate output buffer (worst case = original size or slightly larger) */
        size_t buf_size = tensors[i]->size > 0 ? tensors[i]->size : (size_t)rows * cols * 4;
        if (buf_size > convert_cap) {
            free(convert_out);
            convert_cap = buf_size * 2;
            convert_out = malloc(convert_cap);
        }

        printf("  [%3d/%3d] %-40s %s (%d×%d, %s", i+1, n, tname,
               tclass == CT_TENSOR_SKIP ? "skip" :
               tclass == CT_TENSOR_HEAD ? "head" : format_name,
               rows, cols,
               ct_gguf_type_name(tensors[i]->type));

        sizes[i] = convert_tensor(tensors[i], tdata, out_types[i],
                                   convert_out, tclass);

        data_bufs[i] = malloc(sizes[i]);
        if (data_bufs[i])
            memcpy(data_bufs[i], convert_out, sizes[i]);

        printf(" → %s, %.1f MB)\n",
               ct_gguf_type_name(out_types[i]),
               sizes[i] / (1024.0 * 1024.0));
        total_out += sizes[i];
    }
    free(convert_out);

    printf("\n  Total output size: %.2f GB\n", total_out / (1024.0*1024.0*1024.0));
    printf("\n  Writing GGUF...\n");

    /* Write output */
    bool ok = write_gguf(output_path, src, tensors,
                          (const void**)data_bufs, sizes, out_types, n);

    /* Cleanup */
    for (int i = 0; i < n; i++)
        free(data_bufs[i]);
    free(data_bufs);
    free(tensors);
    free(sizes);
    free(out_types);
    ct_gguf_close(src);

    if (ok) {
        printf("  Done! Output: %s\n", output_path);
        return 0;
    } else {
        fprintf(stderr, "  Failed to write output\n");
        return 1;
    }
}

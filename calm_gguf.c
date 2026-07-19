/**
 * calm_gguf.c — GGUF v3 Tensor Loader
 *
 * Zero-copy mmap-based weight reader. Parses GGUF header, metadata,
 * and tensor info, then provides direct pointers into the mmap'd data.
 *
 * LLM inference по-человечески: mmap + указатели, без копирования.
 */
#define _GNU_SOURCE
#include "calm_gguf.h"
#include "calm_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ═══════════════════════════════════════════════════════════════
 * Helper: read a GGUF key-length (uint32 with uint64 detection)
 * ═══════════════════════════════════════════════════════════════ */

static uint64_t read_key_len(const uint8_t* data, size_t offset, size_t size,
                              size_t* advance) {
    uint32_t len32;
    memcpy(&len32, data + offset, 4);
    /* Перед uint64 стоит uint32 с тем же значением, затем 4 нулевых байта */
    if (offset + 8 <= size) {
        uint64_t full;
        memcpy(&full, data + offset, 8);
        if ((full >> 32) == 0 && len32 > 0) {
            /* uint64 key length */
            *advance = 8;
            return (uint64_t)len32;
        }
    }
    *advance = 4;
    return len32;
}

/* ═══════════════════════════════════════════════════════════════
 * Open and parse GGUF header
 * ═══════════════════════════════════════════════════════════════ */

ct_gguf_context* ct_gguf_open(const char* path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "gguf: cannot open %s\n", path);
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    /* mmap the whole file */
    void* base = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        fprintf(stderr, "gguf: mmap failed for %s\n", path);
        return NULL;
    }

    uint8_t* data = (uint8_t*)base;
    ct_gguf_context* ctx = calloc(1, sizeof(ct_gguf_context));
    if (!ctx) {
        munmap(base, file_size);
        close(fd);
        return NULL;
    }
    ctx->data = (char*)data;
    ctx->size = file_size;
    ctx->fd = fd;

    /* --- Parse header --- */
    size_t pos = 0;

    /* Magic */
    if (pos + 4 > file_size) goto fail;
    uint32_t magic = *(const uint32_t*)(data + pos);
    pos += 4;
    if (magic != CT_GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic 0x%08X\n", magic);
        goto fail;
    }

    /* Version */
    if (pos + 4 > file_size) goto fail;
    ctx->version = *(const uint32_t*)(data + pos);
    pos += 4;

    /* Tensor count + metadata count */
    if (ctx->version == 1) {
        if (pos + 8 > file_size) goto fail;
        ctx->tensor_count = *(const uint32_t*)(data + pos);
        ctx->metadata_count = *(const uint32_t*)(data + pos + 4);
        pos += 8;
    } else {
        if (pos + 16 > file_size) goto fail;
        ctx->tensor_count = *(const uint64_t*)(data + pos);
        ctx->metadata_count = *(const uint64_t*)(data + pos + 8);
        pos += 16;
    }

    /* --- Parse metadata KV pairs --- */
    ctx->metadata.count = 0;
    if (ctx->metadata_count < 1024) { /* sanity */
        ctx->metadata.keys = calloc((size_t)ctx->metadata_count, sizeof(char*));
        ctx->metadata.types = calloc((size_t)ctx->metadata_count, sizeof(uint8_t));
        ctx->metadata.values = calloc((size_t)ctx->metadata_count,
                                      sizeof(*ctx->metadata.values));
    }

    for (uint64_t i = 0; i < ctx->metadata_count; i++) {
        if (pos >= file_size) goto fail;

        /* Key length */
        size_t advance;
        uint64_t key_len = read_key_len(data, pos, file_size - pos, &advance);
        pos += advance;
        if (key_len > 4096 || pos + key_len > file_size) goto fail;

        /* Key string */
        char* key = malloc((size_t)key_len + 1);
        memcpy(key, data + pos, (size_t)key_len);
        key[key_len] = '\0';
        pos += (size_t)key_len;

        /* Value type */
        if (pos + 4 > file_size) { free(key); goto fail; }
        uint32_t val_type = *(const uint32_t*)(data + pos);
        pos += 4;

        /* Store key */
        if (ctx->metadata.count < (int)ctx->metadata_count) {
            int idx = ctx->metadata.count++;
            ctx->metadata.keys[idx] = key;
            ctx->metadata.types[idx] = (uint8_t)val_type;
        } else {
            free(key);
        }

        /* Read value */
        bool ok = true;
        int idx = ctx->metadata.count - 1;
        switch (val_type) {
            case CT_GGUF_VALUE_UINT8:
                if (pos + 1 > file_size) goto fail;
                ctx->metadata.values[idx].v_u8 = data[pos++];
                break;
            case CT_GGUF_VALUE_INT8:
                if (pos + 1 > file_size) goto fail;
                ctx->metadata.values[idx].v_i8 = *(const int8_t*)(data + pos);
                pos += 1;
                break;
            case CT_GGUF_VALUE_UINT16:
                if (pos + 2 > file_size) goto fail;
                ctx->metadata.values[idx].v_u16 = *(const uint16_t*)(data + pos);
                pos += 2;
                break;
            case CT_GGUF_VALUE_INT16:
                if (pos + 2 > file_size) goto fail;
                ctx->metadata.values[idx].v_i16 = *(const int16_t*)(data + pos);
                pos += 2;
                break;
            case CT_GGUF_VALUE_UINT32:
                if (pos + 4 > file_size) goto fail;
                ctx->metadata.values[idx].v_u32 = *(const uint32_t*)(data + pos);
                pos += 4;
                break;
            case CT_GGUF_VALUE_INT32:
                if (pos + 4 > file_size) goto fail;
                ctx->metadata.values[idx].v_i32 = *(const int32_t*)(data + pos);
                pos += 4;
                break;
            case CT_GGUF_VALUE_FLOAT32:
                if (pos + 4 > file_size) goto fail;
                ctx->metadata.values[idx].v_f32 = *(const float*)(data + pos);
                pos += 4;
                break;
            case CT_GGUF_VALUE_BOOL:
                if (pos + 1 > file_size) goto fail;
                ctx->metadata.values[idx].v_bool = data[pos++] != 0;
                break;
            case CT_GGUF_VALUE_STRING: {
                if (pos + 8 > file_size) goto fail;
                uint64_t slen = *(const uint64_t*)(data + pos);
                pos += 8;
                if (slen > file_size - pos) goto fail;
                char* s = malloc((size_t)slen + 1);
                memcpy(s, data + pos, (size_t)slen);
                s[slen] = '\0';
                ctx->metadata.values[idx].v_str = s;
                pos += (size_t)slen;
                break;
            }
            case CT_GGUF_VALUE_ARRAY: {
                if (pos + 12 > file_size) goto fail;
                uint32_t arr_type = *(const uint32_t*)(data + pos);
                uint64_t arr_len = *(const uint64_t*)(data + pos + 4);
                pos += 12;
                ctx->metadata.values[idx].v_str = NULL;

                /* Detect vocabulary arrays by key name */
                const char* k = ctx->metadata.keys[idx];
                int is_vocab_tokens = (k && strcmp(k, "tokenizer.ggml.tokens") == 0
                                        && arr_type == CT_GGUF_VALUE_STRING);
                int is_vocab_scores = (k && strcmp(k, "tokenizer.ggml.scores") == 0
                                        && arr_type == CT_GGUF_VALUE_FLOAT32);
                int is_vocab_types  = (k && strcmp(k, "tokenizer.ggml.token_type") == 0
                                        && arr_type == CT_GGUF_VALUE_INT32);

                if (is_vocab_tokens) {
                    ctx->vocab.n_vocab = (int)arr_len;
                    ctx->vocab.tokens = calloc((size_t)arr_len, sizeof(char*));
                    for (uint64_t j = 0; j < arr_len; j++) {
                        if (pos + 8 > file_size) goto fail;
                        uint64_t slen = *(const uint64_t*)(data + pos);
                        pos += 8;
                        if (slen > file_size - pos) goto fail;
                        char* s = malloc((size_t)slen + 1);
                        memcpy(s, data + pos, (size_t)slen);
                        s[slen] = '\0';
                        ctx->vocab.tokens[j] = s;
                        pos += (size_t)slen;
                    }
                } else if (is_vocab_scores) {
                    ctx->vocab.n_vocab = (int)arr_len;
                    ctx->vocab.scores = calloc((size_t)arr_len, sizeof(float));
                    for (uint64_t j = 0; j < arr_len; j++) {
                        ctx->vocab.scores[j] = *(const float*)(data + pos);
                        pos += 4;
                    }
                } else if (is_vocab_types) {
                    ctx->vocab.n_vocab = (int)arr_len;
                    ctx->vocab.token_types = calloc((size_t)arr_len, sizeof(int));
                    for (uint64_t j = 0; j < arr_len; j++) {
                        ctx->vocab.token_types[j] = *(const int32_t*)(data + pos);
                        pos += 4;
                    }
                } else {
                    /* Skip non-vocab arrays */
                    for (uint64_t j = 0; j < arr_len; j++) {
                        if (pos >= file_size) goto fail;
                        switch (arr_type) {
                            case CT_GGUF_VALUE_UINT8:  case CT_GGUF_VALUE_INT8:  case CT_GGUF_VALUE_BOOL:
                                pos += 1; break;
                            case CT_GGUF_VALUE_UINT16: case CT_GGUF_VALUE_INT16: pos += 2; break;
                            case CT_GGUF_VALUE_UINT32: case CT_GGUF_VALUE_INT32: case CT_GGUF_VALUE_FLOAT32:
                                pos += 4; break;
                            case CT_GGUF_VALUE_UINT64: case CT_GGUF_VALUE_INT64: case CT_GGUF_VALUE_FLOAT64:
                                pos += 8; break;
                            case CT_GGUF_VALUE_STRING: {
                                if (pos + 8 > file_size) goto fail;
                                uint64_t sl = *(const uint64_t*)(data + pos);
                                pos += 8;
                                if (sl > file_size - pos) goto fail;
                                pos += (size_t)sl;
                                break;
                            }
                            default: pos += 4; break;
                        }
                    }
                }
                break;
            }
            case CT_GGUF_VALUE_UINT64:
                if (pos + 8 > file_size) goto fail;
                ctx->metadata.values[idx].v_u64 = *(const uint64_t*)(data + pos);
                pos += 8;
                break;
            case CT_GGUF_VALUE_INT64:
                if (pos + 8 > file_size) goto fail;
                ctx->metadata.values[idx].v_i64 = *(const int64_t*)(data + pos);
                pos += 8;
                break;
            case CT_GGUF_VALUE_FLOAT64:
                if (pos + 8 > file_size) goto fail;
                ctx->metadata.values[idx].v_f64 = *(const double*)(data + pos);
                pos += 8;
                break;
            default:
                pos += 4; /* skip unknown */
                break;
        }
        (void)ok;
    }

    /* --- Parse tensor info --- */
    if (ctx->tensor_count < 4096) {
        ctx->tensors = calloc((size_t)ctx->tensor_count, sizeof(ct_gguf_tensor_info));
    }

    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        if (pos >= file_size) goto fail;

        /* Name length */
        size_t advance;
        uint64_t name_len = read_key_len(data, pos, file_size - pos, &advance);
        pos += advance;
        if (name_len > 256 || pos + name_len > file_size) goto fail;

        char name[256];
        memcpy(name, data + pos, (size_t)name_len);
        name[name_len] = '\0';
        pos += (size_t)name_len;

        /* n_dims (GGUF v3: uint32_t) */
        if (pos + 4 > file_size) goto fail;
        uint32_t n_dims;
        memcpy(&n_dims, data + pos, 4);
        pos += 4;

        /* Dimensions (int64_t[n_dims] - come BEFORE type in GGUF v3!) */
        uint64_t dims[4] = {0};
        for (uint32_t d = 0; d < n_dims && d < 4; d++) {
            if (pos + 8 > file_size) goto fail;
            memcpy(&dims[d], data + pos, 8);
            pos += 8;
        }

        /* Type (ggml_type uint32_t - comes AFTER dims in GGUF v3) */
        if (pos + 4 > file_size) goto fail;
        uint32_t type;
        memcpy(&type, data + pos, 4);
        pos += 4;
        /* Offset from tensor data start */
        if (pos + 8 > file_size) goto fail;
        size_t offset;
        memcpy(&offset, data + pos, 8);
        pos += 8;

        if (i < ctx->tensor_count) {
            ct_gguf_tensor_info* t = &ctx->tensors[i];
            strncpy(t->name, name, sizeof(t->name) - 1);
            t->type = (int)type;
            t->n_dims = n_dims;
            memcpy(t->dims, dims, sizeof(uint64_t) * 4);
            t->offset = offset;
            t->size = ct_gguf_tensor_size(type, (int)n_dims, dims);
        }
    }

    /* Tensor data begins here. GGUF v3 requires 32-byte alignment. */
    ctx->tensor_data_offset = (pos + 31) & ~(size_t)31;

    return ctx;

fail:
    ct_gguf_close(ctx);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════
 * Tensor data access
 * ═══════════════════════════════════════════════════════════════ */

const ct_gguf_tensor_info* ct_gguf_find_tensor(const ct_gguf_context* ctx,
                                                 const char* name) {
    if (!ctx || !name) return NULL;
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    }
    return NULL;
}

const void* ct_gguf_tensor_data(const ct_gguf_context* ctx,
                                 const ct_gguf_tensor_info* t) {
    if (!ctx || !t) return NULL;
    return ctx->data + ctx->tensor_data_offset + t->offset;
}

/* ═══════════════════════════════════════════════════════════════
 * Metadata access
 * ═══════════════════════════════════════════════════════════════ */

const char* ct_gguf_meta_string(const ct_gguf_context* ctx, const char* key) {
    if (!ctx || !key) return NULL;
    for (int i = 0; i < ctx->metadata.count; i++) {
        if (strcmp(ctx->metadata.keys[i], key) == 0 &&
            ctx->metadata.types[i] == CT_GGUF_VALUE_STRING)
            return ctx->metadata.values[i].v_str;
    }
    return NULL;
}

uint64_t ct_gguf_meta_uint(const ct_gguf_context* ctx, const char* key,
                            uint64_t def) {
    if (!ctx || !key) return def;
    for (int i = 0; i < ctx->metadata.count; i++) {
        if (strcmp(ctx->metadata.keys[i], key) != 0) continue;
        switch (ctx->metadata.types[i]) {
            case CT_GGUF_VALUE_UINT8:  return ctx->metadata.values[i].v_u8;
            case CT_GGUF_VALUE_UINT16: return ctx->metadata.values[i].v_u16;
            case CT_GGUF_VALUE_UINT32: return ctx->metadata.values[i].v_u32;
            case CT_GGUF_VALUE_UINT64: return ctx->metadata.values[i].v_u64;
            case CT_GGUF_VALUE_INT32:  return (uint64_t)ctx->metadata.values[i].v_i32;
            case CT_GGUF_VALUE_INT64:  return (uint64_t)ctx->metadata.values[i].v_i64;
        }
    }
    return def;
}

float ct_gguf_meta_float(const ct_gguf_context* ctx, const char* key,
                          float def) {
    if (!ctx || !key) return def;
    for (int i = 0; i < ctx->metadata.count; i++) {
        if (strcmp(ctx->metadata.keys[i], key) != 0) continue;
        switch (ctx->metadata.types[i]) {
            case CT_GGUF_VALUE_FLOAT32: return ctx->metadata.values[i].v_f32;
            case CT_GGUF_VALUE_FLOAT64: return (float)ctx->metadata.values[i].v_f64;
            case CT_GGUF_VALUE_UINT8:   return (float)ctx->metadata.values[i].v_u8;
            case CT_GGUF_VALUE_INT8:    return (float)ctx->metadata.values[i].v_i8;
            case CT_GGUF_VALUE_UINT16:  return (float)ctx->metadata.values[i].v_u16;
            case CT_GGUF_VALUE_INT16:   return (float)ctx->metadata.values[i].v_i16;
            case CT_GGUF_VALUE_UINT32:  return (float)ctx->metadata.values[i].v_u32;
            case CT_GGUF_VALUE_INT32:   return (float)ctx->metadata.values[i].v_i32;
            case CT_GGUF_VALUE_UINT64:  return (float)ctx->metadata.values[i].v_u64;
            case CT_GGUF_VALUE_INT64:   return (float)ctx->metadata.values[i].v_i64;
        }
    }
    return def;
}

const char* ct_gguf_architecture(const ct_gguf_context* ctx) {
    return ct_gguf_meta_string(ctx, "general.architecture");
}

/* ═══════════════════════════════════════════════════════════════
 * Close
 * ═══════════════════════════════════════════════════════════════ */

void ct_gguf_close(ct_gguf_context* ctx) {
    if (!ctx) return;
    if (ctx->data && ctx->size > 0)
        munmap(ctx->data, ctx->size);
    if (ctx->fd >= 0)
        close(ctx->fd);

    /* Free metadata strings */
    for (int i = 0; i < ctx->metadata.count; i++) {
        free(ctx->metadata.keys[i]);
        if (ctx->metadata.types[i] == CT_GGUF_VALUE_STRING)
            free(ctx->metadata.values[i].v_str);
    }
    free(ctx->metadata.keys);
    free(ctx->metadata.types);
    free(ctx->metadata.values);
    /* Free vocabulary */
    if (ctx->vocab.tokens) {
        for (int i = 0; i < ctx->vocab.n_vocab; i++)
            free(ctx->vocab.tokens[i]);
        free(ctx->vocab.tokens);
    }
    free(ctx->vocab.scores);
    free(ctx->vocab.token_types);
    free(ctx->tensors);
    free(ctx);
}

/* ═══════════════════════════════════════════════════════════════
 * Vocabulary access
 * ═══════════════════════════════════════════════════════════════ */

int ct_gguf_load_vocab(ct_gguf_context* ctx) {
    if (!ctx) return -1;
    /* Already loaded if token array is populated */
    if (ctx->vocab.tokens) return 0;
    return -1; /* Not found in GGUF metadata */
}

const char* ct_gguf_vocab_get(const ct_gguf_context* ctx, int id) {
    if (!ctx || !ctx->vocab.tokens) return NULL;
    if (id < 0 || id >= ctx->vocab.n_vocab) return NULL;
    return ctx->vocab.tokens[id];
}

/* ═══════════════════════════════════════════════════════════════
 * Size calculation for tensor types
 * ═══════════════════════════════════════════════════════════════ */

size_t ct_gguf_tensor_size(int type, int n_dims, const uint64_t* dims) {
    if (n_dims < 1 || !dims) return 0;
    uint64_t nelements = 1;
    for (int i = 0; i < n_dims; i++)
        nelements *= dims[i];
    if (nelements == 0) return 0;

    int64_t rows, cols;
    if (n_dims == 1) {
        rows = 1;
        cols = (int64_t)dims[0];
    } else {
        rows = (int64_t)dims[1];  /* output dim */
        cols = (int64_t)dims[0];  /* input dim */
    }

    switch (type) {
        case CT_GGUF_TYPE_F32:
            return (size_t)rows * (size_t)cols * 4;
        case CT_GGUF_TYPE_F16:
            return (size_t)rows * (size_t)cols * 2;
        case CT_GGUF_TYPE_Q8_0: {
            int nb = (int)((cols + 31) / 32);
            return (size_t)rows * nb * CT_SIZEOF_Q8_0;
        }
        case CT_GGUF_TYPE_Q4_0: {
            int nb = (int)((cols + 31) / 32);
            return (size_t)rows * nb * CT_SIZEOF_Q4_0;
        }
        case CT_GGUF_TYPE_Q4_1: {
            int nb = (int)((cols + 31) / 32);
            return (size_t)rows * nb * 20;
        }
        /* Missing standard types */
        case CT_GGUF_TYPE_Q5_0: {
            int nb = (int)((cols + 31) / 32);
            return (size_t)rows * nb * 22;  /* uint16 d + uint8 qh[4] + uint8 qs[16] = 22 */
        }
        case CT_GGUF_TYPE_Q4_K: {
            int nb = (int)((cols + 255) / 256);
            return (size_t)rows * nb * 144; /* uint16 d + uint16 dmin + uint8[12] + uint8[128] = 144 */
        }
        case CT_GGUF_TYPE_Q5_K: {
            int nb = (int)((cols + 255) / 256);
            return (size_t)rows * nb * 178; /* Q5_K: 178 bytes/256-quant block */
        }
        case CT_GGUF_TYPE_Q6_K: {
            int nb = (int)((cols + 255) / 256);
            return (size_t)rows * nb * 210; /* uint16 d + uint8 ql[128] + uint8 qh[64] + int8[16] = 210 */
        }
        case CT_GGUF_TYPE_Q8_K: {
            int nb = (int)((cols + 127) / 128);
            return (size_t)rows * nb * 130; /* uint16 d + uint8 qs[128] = 130, 128 elem/block */
        }
        /* Our custom formats */
        case CT_GGUF_TYPE_BQ1_0: {
            int ng = (int)((cols + 127) / 128);
            return (size_t)rows * ng * CT_SIZEOF_BQ1_0;
        }
        case CT_GGUF_TYPE_TQ1_0: {
            int nb = (int)((cols + 255) / 256);
            return (size_t)rows * nb * CT_SIZEOF_TQ1_0;
        }
        default:
            /* Unknown: estimate 2 bytes per element */
            return (size_t)nelements * 2;
    }
}

const char* ct_gguf_type_name(int type) {
    switch (type) {
        case CT_GGUF_TYPE_F32:   return "F32";
        case CT_GGUF_TYPE_F16:   return "F16";
        case CT_GGUF_TYPE_Q4_0:  return "Q4_0";
        case CT_GGUF_TYPE_Q4_1:  return "Q4_1";
        case CT_GGUF_TYPE_Q5_0:  return "Q5_0";
        case CT_GGUF_TYPE_Q5_1:  return "Q5_1";
        case CT_GGUF_TYPE_Q8_0:  return "Q8_0";
        case CT_GGUF_TYPE_Q8_1:  return "Q8_1";
        case CT_GGUF_TYPE_Q2_K:  return "Q2_K";
        case CT_GGUF_TYPE_Q3_K:  return "Q3_K";
        case CT_GGUF_TYPE_Q4_K:  return "Q4_K";
        case CT_GGUF_TYPE_Q5_K:  return "Q5_K";
        case CT_GGUF_TYPE_Q6_K:  return "Q6_K";
        case CT_GGUF_TYPE_Q8_K:  return "Q8_K";
        case CT_GGUF_TYPE_IQ1_S: return "IQ1_S";
        case CT_GGUF_TYPE_IQ1_M: return "IQ1_M";
        case CT_GGUF_TYPE_IQ2_XXS: return "IQ2_XXS";
        case CT_GGUF_TYPE_IQ2_XS: return "IQ2_XS";
        case CT_GGUF_TYPE_IQ2_S:  return "IQ2_S";
        case CT_GGUF_TYPE_IQ3_XXS: return "IQ3_XXS";
        case CT_GGUF_TYPE_IQ3_XS: return "IQ3_XS";
        case CT_GGUF_TYPE_IQ3_S:  return "IQ3_S";
        case CT_GGUF_TYPE_IQ4_NL: return "IQ4_NL";
        case CT_GGUF_TYPE_IQ4_XS: return "IQ4_XS";
        case CT_GGUF_TYPE_BQ1_0: return "BQ1_0";
        case CT_GGUF_TYPE_TQ1_0: return "TQ1_0";
        default: return "UNKNOWN";
    }
}

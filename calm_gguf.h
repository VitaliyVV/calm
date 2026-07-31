/**
 * calm_gguf.h — GGUF v3 Tensor Loader
 *
 * Zero-copy mmap-based weight reader for GGUF format models.
 * Supports all standard quant formats plus BQ1_0/TQ1_0.
 */
#ifndef CALM_GGUF_H
#define CALM_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF magic */
#define CT_GGUF_MAGIC 0x46554747u  /* "GGUF" */

/* GGUF tensor types */
typedef enum {
    CT_GGUF_TYPE_F32     = 0,
    CT_GGUF_TYPE_F16     = 1,
    CT_GGUF_TYPE_Q4_0    = 2,
    CT_GGUF_TYPE_Q4_1    = 3,
    CT_GGUF_TYPE_Q5_0    = 6,  /* per GGUF spec v3 */
    CT_GGUF_TYPE_Q5_1    = 7,
    CT_GGUF_TYPE_Q8_0    = 8,
    CT_GGUF_TYPE_Q8_1    = 9,
    CT_GGUF_TYPE_Q2_K    = 10,
    CT_GGUF_TYPE_Q3_K    = 11,
    CT_GGUF_TYPE_Q4_K    = 12,
    CT_GGUF_TYPE_Q5_K    = 13,
    CT_GGUF_TYPE_Q6_K    = 14,
    CT_GGUF_TYPE_Q8_K    = 15,
    CT_GGUF_TYPE_IQ1_S   = 26,
    CT_GGUF_TYPE_IQ1_M   = 27,
    CT_GGUF_TYPE_IQ2_XXS = 28,
    CT_GGUF_TYPE_IQ2_XS  = 29,
    CT_GGUF_TYPE_IQ2_S   = 30,
    CT_GGUF_TYPE_IQ3_XXS = 31,
    CT_GGUF_TYPE_IQ3_XS  = 32,
    CT_GGUF_TYPE_IQ3_S   = 33,
    CT_GGUF_TYPE_IQ4_NL  = 34,
    CT_GGUF_TYPE_IQ4_XS  = 35,
    /* Upstream GGUF type IDs (used by models quantized with upstream llama.cpp/ggml) */
    CT_GGUF_TYPE_IQ4_NL_STD = 20,  /* upstream ID for IQ4_NL, needed for ffn_down in Q2_K models */
    /* Custom Calm formats (negative to avoid collision with GGUF) */
    CT_GGUF_TYPE_BQ1_0   = 64,
    CT_GGUF_TYPE_TQ1_0   = 65,
} ct_gguf_tensor_type;

/* GGUF metadata value types */
#define CT_GGUF_VALUE_UINT8   0
#define CT_GGUF_VALUE_INT8    1
#define CT_GGUF_VALUE_UINT16  2
#define CT_GGUF_VALUE_INT16   3
#define CT_GGUF_VALUE_UINT32  4
#define CT_GGUF_VALUE_INT32   5
#define CT_GGUF_VALUE_FLOAT32 6
#define CT_GGUF_VALUE_BOOL    7
#define CT_GGUF_VALUE_STRING  8
#define CT_GGUF_VALUE_ARRAY   9
#define CT_GGUF_VALUE_UINT64  10
#define CT_GGUF_VALUE_INT64   11
#define CT_GGUF_VALUE_FLOAT64 12

/* Tensor info */
typedef struct {
    char name[128];
    int type;              /* ct_gguf_tensor_type */
    uint32_t n_dims;
    uint64_t dims[4];
    size_t offset;         /* offset from start of mmap data */
    size_t size;           /* data size in bytes */
} ct_gguf_tensor_info;

/* ─── Vocabulary ─── */
typedef struct {
    int n_vocab;
    char** tokens;       /* token strings, NULL if not loaded */
    float* scores;       /* token scores, NULL if not available */
    int*   token_types;  /* token types (GGUF token type), NULL if not available */
} ct_gguf_vocab;

/* GGUF file context (mmap-based) */
typedef struct {
    char*  data;           /* mmap base pointer */
    size_t size;           /* file size */
    int    fd;             /* file descriptor */

    /* Header info */
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;

    /* Tensor info array */
    ct_gguf_tensor_info* tensors;

    /* Metadata key-value store (flat array) */
    struct {
        char** keys;
        uint8_t* types;
        union {
            uint8_t  v_u8;
            int8_t   v_i8;
            uint16_t v_u16;
            int16_t  v_i16;
            uint32_t v_u32;
            int32_t  v_i32;
            float    v_f32;
            bool     v_bool;
            uint64_t v_u64;
            int64_t  v_i64;
            double   v_f64;
            char*    v_str;
        }* values;
        int count;
    } metadata;

    /* Tensor data offset (after header + metadata + tensor info) */
    size_t tensor_data_offset;

    /* Vocabulary (loaded from tokenizer.ggml.* arrays during ct_gguf_open) */
    ct_gguf_vocab vocab;
} ct_gguf_context;

/* Open and mmap a GGUF file. Returns NULL on error. */
ct_gguf_context* ct_gguf_open(const char* path);

/* Look up a tensor by name. Returns NULL if not found. */
const ct_gguf_tensor_info* ct_gguf_find_tensor(const ct_gguf_context* ctx,
                                                 const char* name);

/* Get pointer to tensor weight data (zero-copy from mmap) */
const void* ct_gguf_tensor_data(const ct_gguf_context* ctx,
                                 const ct_gguf_tensor_info* t);

/* Get metadata value helpers */
const char* ct_gguf_meta_string(const ct_gguf_context* ctx, const char* key);
uint64_t    ct_gguf_meta_uint(const ct_gguf_context* ctx, const char* key,
                              uint64_t def);
float       ct_gguf_meta_float(const ct_gguf_context* ctx, const char* key,
                              float def);

/* Get the architecture name (general.architecture) */
const char* ct_gguf_architecture(const ct_gguf_context* ctx);

/* Load vocabulary from GGUF metadata (tokenizer.ggml.tokens etc.).
 * Called automatically during ct_gguf_open for the main arrays.
 * Returns 0 on success, -1 if vocab not found. */
int ct_gguf_load_vocab(ct_gguf_context* ctx);

/* Get token string by ID. Returns NULL if out of range or vocab not loaded. */
const char* ct_gguf_vocab_get(const ct_gguf_context* ctx, int id);

/* Token types (matching GGUF tokenizer.ggml.token_type values) */
#define CT_GGUF_TOKEN_TYPE_NORMAL     0
#define CT_GGUF_TOKEN_TYPE_UNKNOWN    1
#define CT_GGUF_TOKEN_TYPE_CONTROL    2
#define CT_GGUF_TOKEN_TYPE_USER_DEFINED 3
#define CT_GGUF_TOKEN_TYPE_UNUSED     4
#define CT_GGUF_TOKEN_TYPE_BYTE       5

/* Close and unmap */
void ct_gguf_close(ct_gguf_context* ctx);

/* Get size in bytes for a tensor type with given dimensions */
size_t ct_gguf_tensor_size(int type, int n_dims, const uint64_t* dims);

/* Convert GGUF tensor type to Calm quant format name */
const char* ct_gguf_type_name(int type);

#ifdef __cplusplus
}
#endif

#endif /* CALM_GGUF_H */

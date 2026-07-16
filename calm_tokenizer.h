/**
 * calm_tokenizer.h — Calm BPE Tokenizer
 *
 * Loads tokenizer metadata from GGUF files and implements
 * byte-level BPE encoding (GPT-2 / LLaMA-3 style) and decoding.
 *
 * Usage:
 *   ct_tokenizer* tok = ct_tokenizer_load(gguf_ctx);
 *   int tokens[1024];
 *   int n = ct_tokenizer_encode(tok, "Hello, world!", tokens, 1024);
 *   char text[4096];
 *   ct_tokenizer_decode(tok, tokens, n, text, sizeof(text));
 *   ct_tokenizer_free(tok);
 */
#ifndef CALM_TOKENIZER_H
#define CALM_TOKENIZER_H

#include "calm_gguf.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Tokenizer type ─── */
typedef enum {
    CT_TOKENIZER_UNKNOWN,
    CT_TOKENIZER_BPE,         /* GPT-2 / LLaMA-3 / Mistral / Qwen-2 */
    CT_TOKENIZER_SENTENCEPIECE, /* LLaMA-1/2, Gemma (TODO) */
} ct_tokenizer_type;

/* ─── BPE merge entry ─── */
typedef struct {
    int left;      /* token ID of left part  */
    int right;     /* token ID of right part */
    int rank;      /* merge priority (0 = highest) */
    int merged_id; /* token ID of the merged token (-1 = unknown) */
} ct_bpe_merge;

/* ─── Tokenizer state ─── */
typedef struct {
    ct_tokenizer_type type;
    int vocab_size;
    int bos_id;
    int eos_id;
    int pad_id;
    int add_bos;
    int add_eos;

    /* Token strings (vocab) */
    char** tokens;
    int*   token_lens;
    float* scores;

    /* BPE merge table */
    ct_bpe_merge* merges;
    int merge_count;
    /* Sorted merge table for binary search */
    int* merge_order;  /* indices into merges[], sorted by (left, right) */

    /* Byte-to-token lookup for single-byte tokens */
    int byte_to_token[256];
} ct_tokenizer;

/* ─── API ─── */

/* Load tokenizer from GGUF metadata. Returns NULL on failure. */
ct_tokenizer* ct_tokenizer_load(ct_gguf_context* gguf);

/* Check if tokenizer was loaded successfully. */
int ct_tokenizer_ready(const ct_tokenizer* tok);

/* Encode text to token IDs. Returns number of tokens, or <0 on error. */
int ct_tokenizer_encode(ct_tokenizer* tok, const char* text,
                        int* tokens, int max_tokens);

/* Decode token IDs to text. Returns length of text written. */
int ct_tokenizer_decode(ct_tokenizer* tok, const int* tokens, int n_tokens,
                        char* text, size_t text_size);

/* Free tokenizer. */
void ct_tokenizer_free(ct_tokenizer* tok);

#ifdef __cplusplus
}
#endif

#endif /* CALM_TOKENIZER_H */

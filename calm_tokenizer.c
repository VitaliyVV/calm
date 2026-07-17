/**
 * calm_tokenizer.c — Calm BPE Tokenizer
 *
 * Byte-level BPE tokenizer (GPT-2 / LLaMA-3 / Mistral style).
 * Loads from GGUF metadata, encodes text → token IDs, decodes back.
 *
 * Pre-tokenization uses a GPT-2-compatible word-split state machine.
 * BPE merge loop applies merges in rank order with restart.
 */
#include "calm_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>
#include <limits.h>

/* ═══════════════════════════════════════════════════════════════
 * Helpers: UTF-8 decode
 * ═══════════════════════════════════════════════════════════════ */

/* Decode one UTF-8 codepoint from s. Returns codepoint, advances *len. */
static unsigned utf8_decode(const char* s, int* len) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80)   { *len = 1; return c; }
    if (c < 0xC0)   { *len = 1; return c; } /* continuation — invalid start */
    if (c < 0xE0)   { *len = 2; return ((unsigned)(c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F); }
    if (c < 0xF0)   { *len = 3; return ((unsigned)(c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F); }
    if (c < 0xF8)   { *len = 4; return ((unsigned)(c & 0x07) << 18) | (((unsigned char)s[1] & 0x3F) << 12) | (((unsigned char)s[2] & 0x3F) << 6) | ((unsigned char)s[3] & 0x3F); }
    *len = 1;
    return c;
}

/* ═══════════════════════════════════════════════════════════════
 * Compare function for merge table (qsort/bsearch by left, right)
 * ═══════════════════════════════════════════════════════════════ */

static int merge_cmp(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    /* We compare the actual merge entries via indices */
    return 0; /* placeholder — re-defined below */
}

static int merge_entry_cmp(const void* a, const void* b) {
    const ct_bpe_merge* ma = (const ct_bpe_merge*)a;
    const ct_bpe_merge* mb = (const ct_bpe_merge*)b;
    if (ma->left != mb->left) return ma->left - mb->left;
    return ma->right - mb->right;
}

/* ═══════════════════════════════════════════════════════════════
 * BPE merge rank lookup (binary search on sorted merge table)
 * ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
 * BPE merge entry lookup (binary search on sorted merge table)
 * ═══════════════════════════════════════════════════════════════ */

static ct_bpe_merge* bpe_find_merge(const ct_tokenizer* tok, int left, int right) {
    ct_bpe_merge key;
    key.left = left;
    key.right = right;
    return (ct_bpe_merge*)bsearch(&key, tok->merges,
                                   (size_t)tok->merge_count,
                                   sizeof(ct_bpe_merge),
                                   merge_entry_cmp);
}

/* ═══════════════════════════════════════════════════════════════
 * BPE merge rank lookup (wrapper returns rank only)
 * ═══════════════════════════════════════════════════════════════ */

static int bpe_find_rank(const ct_tokenizer* tok, int left, int right) {
    ct_bpe_merge* found = bpe_find_merge(tok, left, right);
    if (!found) return -1;
    return found->rank;
}

/* ═══════════════════════════════════════════════════════════════
 * Pre-tokenizer: GPT-2 compatible word split
 *
 * Splits input text into "words" that will be BPE-encoded individually.
 * Keeps spaces attached to following word (GPT-2 convention).
 * Handles contractions ('s, 't, 'm, 'll, 've, 're, 'd).
 * ═══════════════════════════════════════════════════════════════ */

/* Character classes for pre-tokenization */
#define CT_CHAR_WHITE 0
#define CT_CHAR_LETTER 1
#define CT_CHAR_DIGIT 2
#define CT_CHAR_PUNCT 3
#define CT_CHAR_OTHER 4

static int char_class(unsigned cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return CT_CHAR_WHITE;
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
        cp == '_' || cp > 127) return CT_CHAR_LETTER;  /* non-ASCII = letter */
    if (cp >= '0' && cp <= '9') return CT_CHAR_DIGIT;
    if (cp <= 126) return CT_CHAR_PUNCT;  /* ASCII punctuation */
    return CT_CHAR_OTHER;
}

/* Check if a character is a contraction trigger (start of 's, 't, etc.) */
static int is_contr_start(unsigned cp) {
    return cp == '\'' || cp == 0x2019;  /* apostrophe or right single quote */
}

/* Pre-tokenize: split text into words.
 * Returns number of words (up to max_words).
 * The caller must NOT free the word strings — they point into the input text. */
static int pre_tokenize(const char* text, int text_len,
                        const char** words, int* word_lens, int max_words) {
    int nw = 0;
    int i = 0;

    while (i < text_len && nw < max_words) {
        /* Skip leading whitespace (will be prepended to first non-whitespace word) */
        int start = i;

        /* Scan whitespace prefix */
        while (i < text_len && (text[i] == ' ' || text[i] == '\t' ||
                                text[i] == '\n' || text[i] == '\r'))
            i++;

        int ws_len = i - start;
        start = i;

        if (i >= text_len) {
            /* Trailing whitespace only — emit as its own word */
            if (ws_len > 0) {
                words[nw] = text + i - ws_len;
                word_lens[nw] = ws_len;
                nw++;
            }
            break;
        }

        /* Now at a non-whitespace character. Build a word. */
        int word_start = i;
        int has_letter = 0;

        /* Collect the main part of the word */
        while (i < text_len) {
            int cl = char_class((unsigned char)text[i]);

            if (cl == CT_CHAR_LETTER || cl == CT_CHAR_DIGIT) {
                has_letter = 1;
                i++;
            } else if (is_contr_start((unsigned char)text[i])) {
                /* Apostrophe — could be contraction. Check ahead. */
                int saved = i;
                i++;
                /* After apostrophe, check for s, t, m, ll, ve, re, d */
                if (i < text_len && (text[i] == 's' || text[i] == 't' ||
                                     text[i] == 'm' || text[i] == 'd')) {
                    i++;
                    break;
                }
                if (i < text_len && text[i] == 'l' && i + 1 < text_len && text[i+1] == 'l') {
                    i += 2;
                    break;
                }
                if (i < text_len && text[i] == 'v' && i + 1 < text_len && text[i+1] == 'e') {
                    i += 2;
                    break;
                }
                if (i < text_len && text[i] == 'r' && i + 1 < text_len && text[i+1] == 'e') {
                    i += 2;
                    break;
                }
                /* Not a known contraction — treat as punctuation and stop */
                i = saved;
                break;
            } else if (cl == CT_CHAR_PUNCT) {
                /* Punctuation: if we have letters/digits already, stop.
                 * Otherwise include it. */
                if (has_letter) break;
                i++;
                break;
            } else {
                /* Whitespace or other — stop */
                break;
            }
        }

        int word_len = i - word_start;

        /* Emit the word with its whitespace prefix */
        int total_len = ws_len + word_len;
        if (total_len > 0) {
            words[nw] = text + word_start - ws_len;
            word_lens[nw] = total_len;
            nw++;
        }
    }

    return nw;
}

/* ═══════════════════════════════════════════════════════════════
 * BPE encode a single word
 *
 * 1. Convert word bytes to byte-level token IDs
 * 2. Iteratively merge lowest-rank BPE pairs
 * 3. Output final token IDs
 * ═══════════════════════════════════════════════════════════════ */

static int bpe_encode_word(const ct_tokenizer* tok,
                           const char* word, int word_len,
                           int* tokens, int max_tokens) {
    if (word_len == 0) return 0;

    /* Step 1: map bytes to single-byte token IDs */
    int n = 0;
    for (int i = 0; i < word_len; i++) {
        unsigned char b = (unsigned char)word[i];
        int tid = tok->byte_to_token[b];
        if (tid < 0) {
            /* Unknown byte — use replacement token */
            tid = tok->byte_to_token[' '];
            if (tid < 0) return -1;
        }
        if (n >= max_tokens) return -1;
        tokens[n++] = tid;
    }

    if (n < 2) return n;

    /* Step 2: iteratively apply BPE merges */
    int merged = 1;
    while (merged && n > 1) {
        merged = 0;

        /* Find the pair with the lowest merge rank */
        int best_pos = -1;
        int best_rank = INT_MAX;
        ct_bpe_merge* best_entry = NULL;

        for (int i = 0; i < n - 1; i++) {
            ct_bpe_merge* me = bpe_find_merge(tok, tokens[i], tokens[i + 1]);
            if (me && me->rank < best_rank) {
                best_rank = me->rank;
                best_pos = i;
                best_entry = me;
            }
        }

        if (best_pos < 0) break;  /* no more merges */

        /* Use precomputed merged_id from the merge entry.
         * This was populated during tokenizer loading by finding
         * the vocab entry whose string is left_str + right_str. */
        int merged_id = best_entry->merged_id;

        if (merged_id < 0) {
            /* Merged token not found in vocab — stop merging */
            break;
        }

        /* Merge: replace tokens[best_pos] and tokens[best_pos+1] with merged_id */
        tokens[best_pos] = merged_id;
        for (int j = best_pos + 1; j < n - 1; j++)
            tokens[j] = tokens[j + 1];
        n--;
        merged = 1;
    }

    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_tokenizer_load
 * ═══════════════════════════════════════════════════════════════ */

ct_tokenizer* ct_tokenizer_load(ct_gguf_context* gguf) {
    if (!gguf) return NULL;

    ct_tokenizer* tok = (ct_tokenizer*)calloc(1, sizeof(ct_tokenizer));
    if (!tok) return NULL;

    /* Defaults */
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->add_bos = 1;

    /* Read tokenizer model type */
    const char* model_str = ct_gguf_meta_string(gguf, "tokenizer.ggml.model");
    if (!model_str) {
        fprintf(stderr, "tokenizer: no tokenizer.ggml.model in metadata\n");
        goto fail;
    }

    if (strcmp(model_str, "gpt2") == 0 || strcmp(model_str, "bpe") == 0) {
        tok->type = CT_TOKENIZER_BPE;
    } else if (strcmp(model_str, "llama") == 0 || strcmp(model_str, "sentencepiece") == 0) {
        tok->type = CT_TOKENIZER_SENTENCEPIECE;
        fprintf(stderr, "tokenizer: SentencePiece not yet implemented\n");
        goto fail;
    } else {
        fprintf(stderr, "tokenizer: unsupported model '%s'\n", model_str);
        goto fail;
    }

    /* Read vocab size from GGUF */
    /* We scan metadata for tokenizer.ggml.tokens array */
    int vocab_size = 0;
    for (int i = 0; i < gguf->metadata.count; i++) {
        if (strcmp(gguf->metadata.keys[i], "tokenizer.ggml.tokens") == 0 &&
            gguf->metadata.types[i] == CT_GGUF_VALUE_ARRAY) {
            /* Array type — we skipped the elements. Need a different approach. */
        }
        if (strcmp(gguf->metadata.keys[i], "tokenizer.ggml.tokens") == 0) {
            /* The type should be ARRAY(9). For ARRAY, our parser stores
             * v_str = NULL. We need to re-parse. */
        }
    }

    /* ─── Read tokenizer arrays from GGUF metadata ───
     * Our GGUF parser skips array elements. For the tokenizer, we need
     * the actual token strings. We re-parse the metadata section directly. */

    /* Locate the metadata section in the GGUF file */
    /* Header: magic(4) + version(4) + tensor_count(8) + metadata_count(8) */
    size_t pos = 24; /* 4 + 4 + 8 + 8 = 24 */
    if (gguf->version == 1) pos = 12;  /* uint32 counts */

    /* Scan metadata KV pairs to find tokenizer arrays */
    int found_tokens = 0;
    int* token_types = NULL;
    float* token_scores = NULL;

    /* We'll re-parse. Our ct_gguf_context has the raw data pointer. */
    const uint8_t* data = (const uint8_t*)gguf->data;
    size_t file_size = gguf->size;
    size_t meta_pos = pos;

    /* Metadata is already parsed by ct_gguf_open into ctx->metadata.
     * For ARRAY values, we stored v_str = NULL but the raw data is still
     * in the mmap. We need to re-parse the arrays we care about. */

    /* Find the position of each tokenizer metadata: re-scan from start */
    pos = 24; /* header size */
    for (uint64_t mi = 0; mi < gguf->metadata_count; mi++) {
        if (pos >= file_size) break;

        /* Read key length + key */
        size_t adv;
        uint64_t klen = 0;
        if (gguf->version == 1) {
            if (pos + 4 > file_size) break;
            klen = *(const uint32_t*)(data + pos);
            pos += 4;
        } else {
            /* v3: check uint64 key length */
            if (pos + 8 > file_size) break;
            uint64_t full = *(const uint64_t*)(data + pos);
            if ((full >> 32) == 0 && (uint32_t)full > 0) {
                klen = (uint32_t)full;
                pos += 8;
            } else {
                klen = full;
                pos += 8;
            }
        }
        if (klen > 4096 || pos + klen > file_size) break;

        char key[256];
        size_t kl = (size_t)klen;
        if (kl >= sizeof(key)) kl = sizeof(key) - 1;
        memcpy(key, data + pos, kl);
        key[kl] = '\0';
        pos += (size_t)klen;

        /* Value type */
        if (pos + 4 > file_size) break;
        uint32_t vtype = *(const uint32_t*)(data + pos);
        pos += 4;

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == CT_GGUF_VALUE_ARRAY) {
            if (pos + 12 > file_size) break;
            uint32_t arr_type = *(const uint32_t*)(data + pos);
            uint64_t arr_len = *(const uint64_t*)(data + pos + 4);
            pos += 12;

            if (arr_type == 8) { /* STRING array */
                tok->vocab_size = (int)arr_len;
                tok->tokens = (char**)calloc((size_t)arr_len, sizeof(char*));
                tok->token_lens = (int*)calloc((size_t)arr_len, sizeof(int));

                for (uint64_t j = 0; j < arr_len && pos < file_size; j++) {
                    if (pos + 8 > file_size) break;
                    uint64_t slen = *(const uint64_t*)(data + pos);
                    pos += 8;
                    if (slen > file_size - pos) break;
                    tok->tokens[j] = (char*)malloc((size_t)slen + 1);
                    memcpy(tok->tokens[j], data + pos, (size_t)slen);
                    tok->tokens[j][slen] = '\0';
                    tok->token_lens[j] = (int)slen;
                    pos += (size_t)slen;
                }
                found_tokens = 1;
            } else {
                /* Skip array */
                for (uint64_t j = 0; j < arr_len && pos < file_size; j++) {
                    if (arr_type == 4) { pos += 4; }       /* uint32 */
                    else if (arr_type == 6) { pos += 4; }   /* float32 */
                    else if (arr_type == 8) {               /* string */
                        if (pos + 8 > file_size) break;
                        uint64_t sl = *(const uint64_t*)(data + pos);
                        pos += 8;
                        pos += (size_t)sl;
                    } else { pos += 4; }
                }
            }
        } else if (strcmp(key, "tokenizer.ggml.scores") == 0 && vtype == CT_GGUF_VALUE_ARRAY) {
            if (pos + 12 > file_size) break;
            uint32_t arr_type = *(const uint32_t*)(data + pos);
            uint64_t arr_len = *(const uint64_t*)(data + pos + 4);
            pos += 12;

            if (arr_type == 6) { /* FLOAT32 array */
                tok->scores = (float*)calloc((size_t)arr_len, sizeof(float));
                for (uint64_t j = 0; j < arr_len && pos + 4 <= file_size; j++) {
                    tok->scores[j] = *(const float*)(data + pos);
                    pos += 4;
                }
            } else {
                /* Skip */
                size_t elem_size = (arr_type == 6 || arr_type == 4) ? 4 : 8;
                pos += (size_t)(arr_len * elem_size);
            }
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == CT_GGUF_VALUE_ARRAY) {
            if (pos + 12 > file_size) break;
            uint32_t arr_type = *(const uint32_t*)(data + pos);
            uint64_t arr_len = *(const uint64_t*)(data + pos + 4);
            pos += 12;

            if (arr_type == 8) { /* STRING array */
                tok->merge_count = (int)arr_len;
                tok->merges = (ct_bpe_merge*)calloc((size_t)arr_len, sizeof(ct_bpe_merge));

                /* Build length-indexed vocab for fast token lookup by (length, string).
                 * This turns left/right ID scanning from O(vocab) to O(tokens_of_length). */
                int* len_count = NULL;
                int** len_idx = NULL;
                int max_vocab_len = 0;
                int* len_pos = NULL;
                if (tok->vocab_size > 0) {
                    for (int v = 0; v < tok->vocab_size; v++)
                        if (tok->token_lens[v] > max_vocab_len)
                            max_vocab_len = tok->token_lens[v];
                    len_count = (int*)calloc((size_t)(max_vocab_len + 1), sizeof(int));
                    if (len_count) {
                        for (int v = 0; v < tok->vocab_size; v++)
                            len_count[tok->token_lens[v]]++;
                        len_idx = (int**)malloc((size_t)(max_vocab_len + 1) * sizeof(int*));
                        len_pos = (int*)calloc((size_t)(max_vocab_len + 1), sizeof(int));
                        if (len_idx && len_pos) {
                            for (int l = 0; l <= max_vocab_len; l++)
                                if (len_count[l] > 0)
                                    len_idx[l] = (int*)malloc((size_t)len_count[l] * sizeof(int));
                            for (int v = 0; v < tok->vocab_size; v++) {
                                int l = tok->token_lens[v];
                                if (l <= max_vocab_len && len_idx[l])
                                    len_idx[l][len_pos[l]++] = v;
                            }
                        }
                    }
                }

                for (uint64_t j = 0; j < arr_len && pos < file_size; j++) {
                    if (pos + 8 > file_size) break;
                    uint64_t slen = *(const uint64_t*)(data + pos);
                    pos += 8;
                    if (slen > file_size - pos) break;

                    /* Merge string format: "left right" (space-separated) */
                    const char* mstr = (const char*)(data + pos);
                    const char* sep = (const char*)memchr(mstr, ' ', (size_t)slen);
                    tok->merges[j].left  = -1;
                    tok->merges[j].right = -1;

                    if (sep && len_idx) {
                        int left_len = (int)(sep - mstr);
                        const char* right_start = sep + 1;
                        int right_len = (int)(slen - (size_t)(right_start - mstr));

                        /* Fast lookup: only scan tokens with matching length */
                        if (left_len <= max_vocab_len && len_idx[left_len]) {
                            for (int vi = 0; vi < len_count[left_len]; vi++) {
                                int v = len_idx[left_len][vi];
                                if (memcmp(tok->tokens[v], mstr, (size_t)left_len) == 0) {
                                    tok->merges[j].left = v;
                                    break;
                                }
                            }
                        }
                        if (right_len <= max_vocab_len && len_idx[right_len]) {
                            for (int vi = 0; vi < len_count[right_len]; vi++) {
                                int v = len_idx[right_len][vi];
                                if (memcmp(tok->tokens[v], right_start, (size_t)right_len) == 0) {
                                    tok->merges[j].right = v;
                                    break;
                                }
                            }
                        }
                    }
                    tok->merges[j].rank = (int)j;
                    pos += (size_t)slen;
                }

                /* Reuse vocab index for merged_id precomputation.
                 * merged_id = vocab token whose string is left_str + right_str. */
                if (len_idx && tok->merge_count > 0) {
                    for (int j = 0; j < tok->merge_count; j++) {
                        int left_id  = tok->merges[j].left;
                        int right_id = tok->merges[j].right;
                        tok->merges[j].merged_id = -1;
                        if (left_id < 0 || right_id < 0) continue;
                        int ml = tok->token_lens[left_id];
                        int mr = tok->token_lens[right_id];
                        int total_len = ml + mr;
                        if (total_len > max_vocab_len || !len_idx[total_len]) continue;
                        for (int vi = 0; vi < len_count[total_len]; vi++) {
                            int v = len_idx[total_len][vi];
                            if (v == left_id || v == right_id) continue;
                            if (memcmp(tok->tokens[v], tok->tokens[left_id], (size_t)ml) != 0) continue;
                            if (memcmp(tok->tokens[v] + ml, tok->tokens[right_id], (size_t)mr) != 0) continue;
                            tok->merges[j].merged_id = v;
                            break;
                        }
                    }
                }

                /* Cleanup vocab index */
                if (len_idx) {
                    for (int l = 0; l <= max_vocab_len; l++)
                        free(len_idx[l]);
                }
                free(len_idx);
                free(len_pos);
                free(len_count);

                /* Sort merges by (left, right) for binary search */
                qsort(tok->merges, (size_t)tok->merge_count,
                      sizeof(ct_bpe_merge), merge_entry_cmp);
            } else {
                /* Skip */
                for (uint64_t j = 0; j < arr_len && pos < file_size; j++) {
                    if (pos + 8 > file_size) break;
                    uint64_t sl = *(const uint64_t*)(data + pos);
                    pos += 8;
                    pos += (size_t)sl;
                }
            }
        } else if (strcmp(key, "tokenizer.ggml.token_type") == 0 && vtype == CT_GGUF_VALUE_ARRAY) {
            if (pos + 12 > file_size) break;
            uint32_t arr_type = *(const uint32_t*)(data + pos);
            uint64_t arr_len = *(const uint64_t*)(data + pos + 4);
            pos += 12;
            /* Skip array elements — use element size lookup */
            #define SKIP_ELEM_SIZE(t) ( \
                (t) == 0 || (t) == 1 || (t) == 7 ? 1ULL : \
                (t) == 2 || (t) == 3 ? 2ULL : \
                (t) == 4 || (t) == 5 || (t) == 6 ? 4ULL : \
                (t) == 10 || (t) == 11 || (t) == 12 ? 8ULL : 4ULL)
            size_t elem_bytes = (size_t)(arr_len * SKIP_ELEM_SIZE(arr_type));
            if (pos + elem_bytes <= file_size)
                pos += elem_bytes;
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            if (vtype == CT_GGUF_VALUE_UINT32) {
                tok->bos_id = *(const uint32_t*)(data + pos);
                pos += 4;
            } else if (vtype == CT_GGUF_VALUE_INT32) {
                tok->bos_id = *(const int32_t*)(data + pos);
                pos += 4;
            } else {
                pos += 4;
            }
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            if (vtype == CT_GGUF_VALUE_UINT32) {
                tok->eos_id = *(const uint32_t*)(data + pos);
                pos += 4;
            } else if (vtype == CT_GGUF_VALUE_INT32) {
                tok->eos_id = *(const int32_t*)(data + pos);
                pos += 4;
            } else {
                pos += 4;
            }
        } else if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0) {
            if (vtype == CT_GGUF_VALUE_BOOL) {
                tok->add_bos = data[pos++] != 0;
            } else {
                pos += 1;
            }
        } else {
            /* Skip value — use our standard parser path.
             * But since we're re-parsing, we need to skip.
             * For anything that's not an array, skip by type. */
            switch (vtype) {
                case 0: case 1: pos += 1; break;
                case 2: case 3: pos += 2; break;
                case 4: case 5: case 6: pos += 4; break;
                case 7: pos += 1; break;
                case 8: { uint64_t sl; memcpy(&sl, data + pos, 8); pos += 8 + sl; break; }
                case 10: case 11: case 12: pos += 8; break;
                default: pos += 4; break;
            }
        }
    }

    if (!found_tokens || !tok->tokens) {
        fprintf(stderr, "tokenizer: no tokenizer.ggml.tokens array found\n");
        goto fail;
    }

    /* Build byte-to-token lookup (GPT-2 byte encoding scheme).
     *
     * GPT-2 maps each byte 0x00-0xFF to a Unicode codepoint:
     *   - 0x21-0x7E (printable ASCII): map to themselves
     *   - 0x00-0x20: map to U+0100..U+0120 (33 codepoints)
     *   - 0x7F-0xFF: map to U+0121..U+01A1 (129 codepoints)
     *
     * For each byte, we build the expected UTF-8 string and scan
     * the full vocab for a matching token.
     */
    static const uint16_t byte_to_cp[256] = {
        /* 0x00-0x20 → U+0100..U+0120 */
        0x0100,0x0101,0x0102,0x0103,0x0104,0x0105,0x0106,0x0107,
        0x0108,0x0109,0x010A,0x010B,0x010C,0x010D,0x010E,0x010F,
        0x0110,0x0111,0x0112,0x0113,0x0114,0x0115,0x0116,0x0117,
        0x0118,0x0119,0x011A,0x011B,0x011C,0x011D,0x011E,0x011F,
        0x0120,  /* 0x20 = space → U+0120 'Ġ' */
        /* 0x21-0x7E → map to themselves */
        0x0021,0x0022,0x0023,0x0024,0x0025,0x0026,0x0027,0x0028,
        0x0029,0x002A,0x002B,0x002C,0x002D,0x002E,0x002F,0x0030,
        0x0031,0x0032,0x0033,0x0034,0x0035,0x0036,0x0037,0x0038,
        0x0039,0x003A,0x003B,0x003C,0x003D,0x003E,0x003F,0x0040,
        0x0041,0x0042,0x0043,0x0044,0x0045,0x0046,0x0047,0x0048,
        0x0049,0x004A,0x004B,0x004C,0x004D,0x004E,0x004F,0x0050,
        0x0051,0x0052,0x0053,0x0054,0x0055,0x0056,0x0057,0x0058,
        0x0059,0x005A,0x005B,0x005C,0x005D,0x005E,0x005F,0x0060,
        0x0061,0x0062,0x0063,0x0064,0x0065,0x0066,0x0067,0x0068,
        0x0069,0x006A,0x006B,0x006C,0x006D,0x006E,0x006F,0x0070,
        0x0071,0x0072,0x0073,0x0074,0x0075,0x0076,0x0077,0x0078,
        0x0079,0x007A,0x007B,0x007C,0x007D,0x007E,
        /* 0x7F-0xFF → U+0121..U+01A1 */
        0x0121,0x0122,0x0123,0x0124,0x0125,0x0126,0x0127,0x0128,
        0x0129,0x012A,0x012B,0x012C,0x012D,0x012E,0x012F,0x0130,
        0x0131,0x0132,0x0133,0x0134,0x0135,0x0136,0x0137,0x0138,
        0x0139,0x013A,0x013B,0x013C,0x013D,0x013E,0x013F,0x0140,
        0x0141,0x0142,0x0143,0x0144,0x0145,0x0146,0x0147,0x0148,
        0x0149,0x014A,0x014B,0x014C,0x014D,0x014E,0x014F,0x0150,
        0x0151,0x0152,0x0153,0x0154,0x0155,0x0156,0x0157,0x0158,
        0x0159,0x015A,0x015B,0x015C,0x015D,0x015E,0x015F,0x0160,
        0x0161,0x0162,0x0163,0x0164,0x0165,0x0166,0x0167,0x0168,
        0x0169,0x016A,0x016B,0x016C,0x016D,0x016E,0x016F,0x0170,
        0x0171,0x0172,0x0173,0x0174,0x0175,0x0176,0x0177,0x0178,
        0x0179,0x017A,0x017B,0x017C,0x017D,0x017E,0x017F,0x0180,
        0x0181,0x0182,0x0183,0x0184,0x0185,0x0186,0x0187,0x0188,
        0x0189,0x018A,0x018B,0x018C,0x018D,0x018E,0x018F,0x0190,
        0x0191,0x0192,0x0193,0x0194,0x0195,0x0196,0x0197,0x0198,
        0x0199,0x019A,0x019B,0x019C,0x019D,0x019E,0x019F,0x01A0,
        0x01A1,
    };

    for (int b = 0; b < 256; b++)
        tok->byte_to_token[b] = -1;

    /* Encode each byte's codepoint as UTF-8 and scan vocab */
    for (int b = 0; b < 256; b++) {
        unsigned cp = byte_to_cp[b];
        uint8_t utf8[8];
        int utf8_len;
        if (cp < 0x80) {
            utf8[0] = (uint8_t)cp;
            utf8_len = 1;
        } else if (cp < 0x800) {
            utf8[0] = 0xC0 | (cp >> 6);
            utf8[1] = 0x80 | (cp & 0x3F);
            utf8_len = 2;
        } else {
            utf8[0] = 0xE0 | (cp >> 12);
            utf8[1] = 0x80 | ((cp >> 6) & 0x3F);
            utf8[2] = 0x80 | (cp & 0x3F);
            utf8_len = 3;
        }
        for (int v = 0; v < tok->vocab_size; v++) {
            if (tok->token_lens[v] == utf8_len &&
                memcmp(tok->tokens[v], utf8, (size_t)utf8_len) == 0) {
                tok->byte_to_token[b] = v;
                break;
            }
        }
    }

    return tok;

fail:
    ct_tokenizer_free(tok);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_tokenizer_ready
 * ═══════════════════════════════════════════════════════════════ */

int ct_tokenizer_ready(const ct_tokenizer* tok) {
    return tok && tok->tokens && tok->vocab_size > 0;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_tokenizer_encode
 * ═══════════════════════════════════════════════════════════════ */

int ct_tokenizer_encode(ct_tokenizer* tok, const char* text,
                        int* tokens, int max_tokens) {
    if (!tok || !tok->tokens || !text || !tokens || max_tokens < 1)
        return -1;

    int text_len = (int)strlen(text);

    /* Add BOS token */
    int total = 0;
    if (tok->add_bos && tok->bos_id >= 0) {
        tokens[total++] = tok->bos_id;
    }

    /* ── Special token scanning ──
     * Scan the input for known special token strings before pre-tokenization.
     * These tokens must be emitted as single token IDs, not split by BPE.
     * Format: { string_literal, token_id }
     * This list should be populated from GGUF metadata ideally, but for now
     * we hardcode the common Qwen2 special tokens. */
    int n_special = 3;
    struct { const char* str; int id; } specials[] = {
        {"<|im_start|>", 151644},
        {"<|im_end|>", 151645},
        {"<|endoftext|>", 151643},
    };

    int pos = 0;
    while (pos < text_len && total < max_tokens) {
        /* Check for special token match at current position */
        int matched = 0;
        for (int s = 0; s < n_special; s++) {
            int slen = (int)strlen(specials[s].str);
            if (pos + slen <= text_len &&
                memcmp(text + pos, specials[s].str, (size_t)slen) == 0) {
                tokens[total++] = specials[s].id;
                pos += slen;
                matched = 1;
                break;
            }
        }
        if (matched) continue;

        /* Not a special token. Find the next special token boundary or end. */
        int start = pos;
        while (pos < text_len) {
            int is_special = 0;
            for (int s = 0; s < n_special; s++) {
                int slen = (int)strlen(specials[s].str);
                if (pos + slen <= text_len &&
                    memcmp(text + pos, specials[s].str, (size_t)slen) == 0) {
                    is_special = 1;
                    break;
                }
            }
            if (is_special) break;
            pos++;
        }

        /* Encode the non-special segment with BPE */
        int seg_len = pos - start;
        if (seg_len > 0) {
            const char* words[4096];
            int word_lens[4096];
            int nw = pre_tokenize(text + start, seg_len, words, word_lens, 4096);
            if (nw < 0) continue;

            int word_tokens[4096];
            for (int w = 0; w < nw && total < max_tokens; w++) {
                int nt = bpe_encode_word(tok, words[w], word_lens[w], word_tokens,
                                          max_tokens - total < 4096 ? max_tokens - total : 4096);
                if (nt < 0) continue;
                for (int i = 0; i < nt && total < max_tokens; i++)
                    tokens[total++] = word_tokens[i];
            }
        }
    }

    return total;
}

/* ═══════════════════════════════════════════════════════════════
 * GPT-2 byte-decoding post-processor
 *
 * GPT-2 byte encoding maps bytes 0x00-0xFF to Unicode codepoints:
 *   0x00-0x20 → U+0100-U+0120  (control chars + space)
 *   0x21-0x7E → self           (printable ASCII, single byte)
 *   0x7F-0xFF → U+0121-U+01A1  (DEL + extended ASCII)
 *
 * During decoding, token strings contain the UTF-8 form of these
 * codepoints. This function converts them back to raw bytes so
 * that spaces show as ' ', newlines as newlines, etc.
 * ═══════════════════════════════════════════════════════════════ */

static void gpt2_decode_bytes(char* text, size_t* len) {
    size_t r = 0, w = 0;
    while (r < *len) {
        unsigned char c = (unsigned char)text[r];

        /* Self-mapped ASCII 0x21-0x7E — output as-is */
        if (c >= 0x21 && c <= 0x7E) {
            text[w++] = c;
            r++;
            continue;
        }

        /* Possible byte-encoded UTF-8: two-byte sequence 0xC4-0xC7 + 0x80-0xBF */
        if (c >= 0xC4 && c <= 0xC7 && r + 1 < *len) {
            unsigned char c2 = (unsigned char)text[r + 1];
            if (c2 >= 0x80 && c2 <= 0xBF) {
                /* Decode the UTF-8 codepoint:
                 *   0xC4 xx → U+0100 | (xx & 3F)
                 *   0xC5 xx → U+0140 | (xx & 3F)
                 *   0xC6 xx → U+0180 | (xx & 3F)
                 *   0xC7 xx → U+01C0 | (xx & 3F)
                 *   ⇒ cp = ((c & 7) << 6) | (c2 & 3F) | 0x0100 */
                unsigned cp = ((unsigned)(c & 7) << 6) | (unsigned)(c2 & 0x3F);
                cp |= 0x0100;

                if (cp <= 0x0120) {
                    /* Byte 0x00-0x20 (controls + space) */
                    text[w++] = (unsigned char)(cp - 0x0100);
                    r += 2;
                    continue;
                }
                if (cp >= 0x0121 && cp <= 0x01A1) {
                    /* Byte 0x7F-0xFF (DEL + extended) */
                    text[w++] = (unsigned char)(cp - 0x00A2); /* = cp - 0x0121 + 0x7F */
                    r += 2;
                    continue;
                }
                /* Otherwise it's a legitimate Unicode codepoint outside
                 * the GPT-2 byte encoding range — pass through */
            }
        }

        /* Any remaining byte: NUL, or something unexpected.
         * Replace NUL with space to avoid truncation, pass others through. */
        text[w++] = c == '\0' ? ' ' : c;
        r++;
    }
    text[w] = '\0';
    *len = w;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_tokenizer_decode
 * ═══════════════════════════════════════════════════════════════ */

int ct_tokenizer_decode(ct_tokenizer* tok, const int* tokens, int n_tokens,
                        char* text, size_t text_size) {
    if (!tok || !tok->tokens || !tokens || !text || text_size == 0)
        return -1;

    size_t pos = 0;
    for (int i = 0; i < n_tokens && pos < text_size; i++) {
        int tid = tokens[i];
        if (tid < 0 || tid >= tok->vocab_size) continue;
        int len = tok->token_lens[tid];
        if (len <= 0) continue;
        if (pos + (size_t)len >= text_size) {
            /* Truncate */
            size_t copy = text_size - pos - 1;
            memcpy(text + pos, tok->tokens[tid], copy);
            pos += copy;
            break;
        }
        memcpy(text + pos, tok->tokens[tid], (size_t)len);
        pos += (size_t)len;
    }
    text[pos] = '\0';

    /* Post-process: convert GPT-2 byte encoding back to raw bytes */
    gpt2_decode_bytes(text, &pos);

    return (int)pos;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_tokenizer_decode_single
 * ═══════════════════════════════════════════════════════════════ */

int ct_tokenizer_decode_single(ct_tokenizer* tok, int token,
                                char* text, size_t text_size) {
    if (!tok || !text || text_size == 0) return -1;
    if (token < 0 || token >= tok->vocab_size) {
        text[0] = '\0';
        return 0;
    }
    int len = tok->token_lens[token];
    if (len <= 0) {
        text[0] = '\0';
        return 0;
    }
    size_t copy = (size_t)len < text_size - 1 ? (size_t)len : text_size - 1;
    memcpy(text, tok->tokens[token], copy);
    text[copy] = '\0';
    size_t pos = copy;
    gpt2_decode_bytes(text, &pos);
    return (int)pos;
}

/* ═══════════════════════════════════════════════════════════════
 * ct_tokenizer_free
 * ═══════════════════════════════════════════════════════════════ */

void ct_tokenizer_free(ct_tokenizer* tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (int i = 0; i < tok->vocab_size; i++)
            free(tok->tokens[i]);
        free(tok->tokens);
    }
    free(tok->token_lens);
    free(tok->scores);
    free(tok->merges);
    free(tok->merge_order);
    free(tok);
}

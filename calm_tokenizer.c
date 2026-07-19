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
             * But since we're re-parsing, we need to skip all types. */
            switch (vtype) {
                case 0: case 1: pos += 1; break;
                case 2: case 3: pos += 2; break;
                case 4: case 5: case 6: pos += 4; break;
                case 7: pos += 1; break;
                case 8: { uint64_t sl; memcpy(&sl, data + pos, 8); pos += 8 + sl; break; }
                case 9: {
                    /* ARRAY: read arr_type + arr_len, then skip elements */
                    if (pos + 12 > file_size) break;
                    uint32_t arr_type = *(const uint32_t*)(data + pos);
                    uint64_t arr_len = *(const uint64_t*)(data + pos + 4);
                    pos += 12;
                    for (uint64_t j = 0; j < arr_len && pos < file_size; j++) {
                        if (arr_type == 8) { /* STRING array */
                            if (pos + 8 > file_size) break;
                            uint64_t slen = *(const uint64_t*)(data + pos);
                            pos += 8;
                            if (slen > file_size - pos) break;
                            pos += (size_t)slen;
                        } else if (arr_type == 4 || arr_type == 5 || arr_type == 6) {
                            pos += 4;
                        } else if (arr_type == 2 || arr_type == 3) {
                            pos += 2;
                        } else if (arr_type == 0 || arr_type == 1 || arr_type == 7) {
                            pos += 1;
                        } else {
                            pos += 4;
                        }
                    }
                    break;
                }
                case 10: case 11: case 12: pos += 8; break;
                default: pos += 4; break;
            }
        }
    }

    if (!found_tokens || !tok->tokens) {
        /* Fallback: use vocab already loaded by ct_gguf_open */
        if (gguf->vocab.tokens && gguf->vocab.n_vocab > 0) {
            tok->vocab_size = gguf->vocab.n_vocab;
            tok->tokens = calloc((size_t)gguf->vocab.n_vocab, sizeof(char*));
            tok->token_lens = calloc((size_t)gguf->vocab.n_vocab, sizeof(int));
            if (tok->tokens && tok->token_lens) {
                for (int i = 0; i < gguf->vocab.n_vocab; i++) {
                    if (gguf->vocab.tokens[i]) {
                        tok->tokens[i] = strdup(gguf->vocab.tokens[i]);
                        tok->token_lens[i] = (int)strlen(gguf->vocab.tokens[i]);
                    }
                }
                found_tokens = 1;
            }
        }
        if (!found_tokens) {
            fprintf(stderr, "tokenizer: no tokenizer.ggml.tokens array found\n");
            goto fail;
        }
    }

    /* Build byte-to-token lookup from actual vocab.
     *
     * Scans the first min(256, vocab_size) tokens and decodes each
     * to determine which byte it represents. Handles both GPT-2 and
     * Llama byte encoding schemes automatically:
     *
     *   GPT-2: 0x00-0x20 → U+0100-U+0120    (cp - 0x0100)
     *          0x21-0x7E → direct             (cp as-is)
     *          0x7F-0xFF → U+0121-U+01A1     (cp - 0x00A2)
     *
     *   Llama: 0x00-0x20 → U+0100-U+0120    (same as GPT-2)
     *          0x21-0x7E → direct             (same as GPT-2)
     *          0x7F-0xA1 → U+0121-U+0143     (cp - 0x00A2)
     *          0xA2-0xFF → U+00A2-U+00FF     (cp as-is, identity)
     */
    for (int b = 0; b < 256; b++)
        tok->byte_to_token[b] = -1;

    int n_byte_tokens = tok->vocab_size < 256 ? tok->vocab_size : 256;
    for (int v = 0; v < n_byte_tokens; v++) {
        if (tok->token_lens[v] <= 0) continue;

        /* Decode the token string as UTF-8 */
        int ulen;
        unsigned cp = utf8_decode(tok->tokens[v], &ulen);
        if (ulen != tok->token_lens[v]) continue;  /* multi-codepoint? skip */

        int byte_val = -1;
        if (cp >= 0x21 && cp <= 0x7E && ulen == 1) {
            byte_val = cp;                                     /* direct ASCII */
        } else if (cp >= 0x100 && cp <= 0x120 && ulen == 2) {
            byte_val = (int)(cp - 0x0100);                     /* bytes 0x00-0x20 */
        } else if (cp >= 0x121 && cp <= 0x1A1 && ulen == 2) {
            byte_val = (int)(cp - 0x00A2);                     /* bytes 0x7F-0xFF */
        } else if (cp >= 0xA1 && cp <= 0xFF && ulen == 2) {
            byte_val = (int)cp;                                /* Llama identity */
        } else if (cp < 0x21 && ulen == 1) {
            byte_val = (int)cp;                                /* raw control byte */
        }

        if (byte_val >= 0 && byte_val < 256)
            tok->byte_to_token[byte_val] = v;
    }

    /* ── Discover special tokens from vocab ──
     * Scan for tokens matching <|...|> and store for use during encode.
     * These must be emitted as single tokens, not split by BPE. */
    tok->specials = NULL;
    tok->n_specials = 0;
    for (int v = 0; v < tok->vocab_size; v++) {
        if (tok->token_lens[v] <= 0) continue;
        const char* s = tok->tokens[v];
        int slen = tok->token_lens[v];
        /* Match <|...|> pattern with len >= 5 (<||> min) */
        if (slen >= 5 && s[0] == '<' && s[1] == '|' && s[slen-2] == '|' && s[slen-1] == '>') {
            void* p = realloc(tok->specials, sizeof(*tok->specials) * (size_t)(tok->n_specials + 1));
            if (!p) continue;
            tok->specials = p;
            tok->specials[tok->n_specials].str = strndup(s, (size_t)slen);
            tok->specials[tok->n_specials].id  = v;
            tok->specials[tok->n_specials].len = slen;
            tok->n_specials++;
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
     * Special token list is populated from vocab during tokenizer load
     * (tokens matching <|...|> pattern, e.g. Llama <|start_header_id|>,
     * Qwen <|im_start|>, etc.). */
    int n_special = tok->n_specials;

    int pos = 0;
    while (pos < text_len && total < max_tokens) {
        /* Check for special token match at current position */
        int matched = 0;
        for (int s = 0; s < n_special; s++) {
            int slen = tok->specials[s].len;
            if (pos + slen <= text_len &&
                memcmp(text + pos, tok->specials[s].str, (size_t)slen) == 0) {
                tokens[total++] = tok->specials[s].id;
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
                int slen = tok->specials[s].len;
                if (pos + slen <= text_len &&
                    memcmp(text + pos, tok->specials[s].str, (size_t)slen) == 0) {
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

        /* Llama-style byte encoding: two-byte 0xC2-0xC3 + 0x80-0xBF
         * Bytes 0xA2-0xFF → codepoints U+00A2-U+00FF (identity mapping)
         * The UTF-8 of these codepoints starts with 0xC2-0xC3, which is
         * outside the GPT-2 0xC4-0xC7 range checked above.
         * Output the decoded codepoint as a single byte (identity).
         * This is SAFE for GPT-2 models too — 0xC2-0xC3 never appears
         * in GPT-2 byte-encoded token strings. */
        if (c >= 0xC2 && c <= 0xC3 && r + 1 < *len) {
            unsigned char c2 = (unsigned char)text[r + 1];
            if (c2 >= 0x80 && c2 <= 0xBF) {
                unsigned cp = ((unsigned)(c & 0x1F) << 6) | (unsigned)(c2 & 0x3F);
                /* Identity mapping: cp == byte value for 0x80-0xFF */
                text[w++] = (unsigned char)cp;
                r += 2;
                continue;
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
    if (tok->specials) {
        for (int i = 0; i < tok->n_specials; i++)
            free(tok->specials[i].str);
        free(tok->specials);
    }
    free(tok);
}

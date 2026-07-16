/**
 * test_tokenizer.c — Standalone BPE tokenizer test.
 *
 * Creates a synthetic GPT-2-style tokenizer with known vocab + merges
 * and tests encode/decode roundtrip.
 */
#include "calm_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int merge_cmp(const void* a, const void* b);

/* ─── Build a minimal test tokenizer manually ─── */
static ct_tokenizer* make_test_tokenizer(void) {
    ct_tokenizer* tok = (ct_tokenizer*)calloc(1, sizeof(ct_tokenizer));
    if (!tok) return NULL;

    tok->type = CT_TOKENIZER_BPE;
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->add_bos = 0;  /* no BOS for testing */
    tok->vocab_size = 10;

    /* Vocab: single bytes (0-255) plus merged tokens
     * Tokens 0-9: a b c d e space . !
     * For simplicity, tokens 0-9 map to single characters.
     * In real GPT-2, bytes 0-255 are tokens 0-255.
     */
    tok->tokens = (char**)calloc(10, sizeof(char*));
    tok->token_lens = (int*)calloc(10, sizeof(int));
    const char* vocab_str[] = {"a", "b", "c", "d", "e", " ", ".", "!", "ab", "bc"};
    for (int i = 0; i < 10; i++) {
        tok->tokens[i] = strdup(vocab_str[i]);
        tok->token_lens[i] = (int)strlen(vocab_str[i]);
    }

    /* Byte-to-token mapping: map each token's single byte */
    for (int i = 0; i < 256; i++)
        tok->byte_to_token[i] = -1;
    for (int i = 0; i < 10; i++) {
        if (tok->token_lens[i] == 1) {
            tok->byte_to_token[(unsigned char)tok->tokens[i][0]] = i;
        }
    }
    /* Ensure common byte tokens are available */
    tok->byte_to_token[(unsigned char)'a'] = 0;
    tok->byte_to_token[(unsigned char)'b'] = 1;
    tok->byte_to_token[(unsigned char)'c'] = 2;
    tok->byte_to_token[(unsigned char)'d'] = 3;
    tok->byte_to_token[(unsigned char)'e'] = 4;
    tok->byte_to_token[(unsigned char)' '] = 5;
    tok->byte_to_token[(unsigned char)'.'] = 6;
    tok->byte_to_token[(unsigned char)'!'] = 7;

    /* Merges: "ab" = token 8 (rank 0), "bc" = token 9 (rank 1) */
    tok->merge_count = 2;
    tok->merges = (ct_bpe_merge*)calloc(2, sizeof(ct_bpe_merge));
    tok->merges[0].left = 0;  /* a */
    tok->merges[0].right = 1; /* b */
    tok->merges[0].rank = 0;
    tok->merges[1].left = 1;  /* b */
    tok->merges[1].right = 2; /* c */
    tok->merges[1].rank = 1;

    /* Sort by (left, right) */
    qsort(tok->merges, 2, sizeof(ct_bpe_merge), merge_cmp);

    return tok;
}

static int merge_cmp(const void* a, const void* b) {
    const ct_bpe_merge* ma = (const ct_bpe_merge*)a;
    const ct_bpe_merge* mb = (const ct_bpe_merge*)b;
    if (ma->left != mb->left) return ma->left - mb->left;
    return ma->right - mb->right;
}

int main(void) {
    int errors = 0;

    ct_tokenizer* tok = make_test_tokenizer();
    if (!tok) {
        printf("FAIL: couldn't create tokenizer\n");
        return 1;
    }

    printf("Tokenizer: %d tokens, %d merges\n", tok->vocab_size, tok->merge_count);
    printf("  Tokens: ");
    for (int i = 0; i < tok->vocab_size; i++)
        printf("'%s' ", tok->tokens[i]);
    printf("\n");

    /* ─── Test 1: Basic decode ─── */
    {
        int ids[] = {0, 1, 2};  /* a b c */
        char text[64];
        int len = ct_tokenizer_decode(tok, ids, 3, text, sizeof(text));
        printf("Test 1 (decode [a,b,c]): \"%s\" (len=%d)\n", text, len);
        if (strcmp(text, "abc") != 0) {
            printf("  FAIL: expected 'abc'\n");
            errors++;
        } else {
            printf("  PASS\n");
        }
    }

    /* ─── Test 2: Decode with spaces ─── */
    {
        int ids[] = {0, 5, 1, 5, 2};  /* a space b space c */
        char text[64];
        ct_tokenizer_decode(tok, ids, 5, text, sizeof(text));
        printf("Test 2 (decode [a,sp,b,sp,c]): \"%s\"\n", text);
        if (strcmp(text, "a b c") != 0) {
            printf("  FAIL: expected 'a b c'\n");
            errors++;
        } else {
            printf("  PASS\n");
        }
    }

    /* ─── Test 3: BPE encode ─── */
    {
        int tokens[16];
        int n = ct_tokenizer_encode(tok, "abc", tokens, 16);
        printf("Test 3 (encode 'abc'): %d tokens:", n);
        for (int i = 0; i < n; i++) printf(" '%s'", tok->tokens[tokens[i]]);
        printf("\n");

        /* Expected: "ab" (token 8) + "c" (token 2), or "a" + "bc" (token 9)
         * "ab" has rank 0, "bc" has rank 1.
         * Starting with [a,b,c]:
         *   Step 1: find pair with lowest rank.
         *     (a,b) = rank 0, (b,c) = rank 1 → merge a,b → [ab, c]
         *   Step 2: find pairs in [ab, c]: only (ab,c) not in merges → stop
         * Result: [ab, c] = [8, 2]
         */
        if (n == 2 && tokens[0] == 8 && tokens[1] == 2) {
            printf("  PASS\n");
        } else {
            printf("  FAIL: expected [8,2] (ab, c)\n");
            errors++;
        }
    }

    /* ─── Test 4: Decode after encode roundtrip ─── */
    {
        char input[] = "abc abc";
        int tokens[16];
        int n = ct_tokenizer_encode(tok, input, tokens, 16);
        char output[64];
        ct_tokenizer_decode(tok, tokens, n, output, sizeof(output));
        printf("Test 4 (roundtrip '%s'): '%s' (%d tokens)\n", input, output, n);
        if (strcmp(input, output) == 0) {
            printf("  PASS\n");
        } else {
            printf("  FAIL: roundtrip mismatch\n");
            errors++;
        }
    }

    /* ─── Test 5: Pre-tokenizer with contractions ─── */
    {
        /* "don't" should be recognized with contraction */
        int tokens[16];
        const char* text = "don't";
        int n = ct_tokenizer_encode(tok, text, tokens, 16);
        printf("Test 5 (encode '%s'): %d tokens\n", text, n);

        if (n > 0) {
            printf("  Tokens:");
            for (int i = 0; i < n; i++) printf(" '%s'", tok->tokens[tokens[i]]);
            printf("\n");
            printf("  PASS (no crash)\n");
        } else {
            printf("  FAIL: no tokens\n");
            errors++;
        }
    }

    /* ─── Summary ─── */
    printf("\n=== %d / %d tests passed ===\n",
           errors == 0 ? 5 : 5 - errors, 5);

    ct_tokenizer_free(tok);
    return errors > 0 ? 1 : 0;
}

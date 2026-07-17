# Calm Architecture

**Версия:** 0.2
**Дата:** 16 июля 2026

---

## 1. Module Structure (Current)

```
calm                      # Main binary — CLI + generation loop
├── calm.c                #   CLI, inference orchestration, generation loop
├── calm.h                #   Public API (CalmRuntime, CalmModel, params)

├── calm_infer.c          #   Native transformer inference
│                           #   RMS norm, RoPE, SiLU, attention, SwiGLU FFN
├── calm_gguf.c           #   GGUF format parser (tensors, metadata, config)
├── calm_tokenizer.c      #   BPE tokenizer (GPT-2 byte-level, merged-id precompute)
├── calm_quant.c          #   Quantized type math (FP32, Q4_0, Q8_0, BQ1_0, TQ1_0)

├── calm_server.c         #   HTTP API server (POSIX sockets, /v1/completions)
├── calm_tools.c          #   Function/tool calling (Qwen2.5 format, JSON parser)

└── calm_convert.c        # Standalone converter: FP32/Q8_0 → BQ1_0/TQ1_0
├── calm_vulkan.c         #   Vulkan GPU compute backend (Q8_0 matmul on Adreno)
├── calm_vulkan.h         #   Vulkan backend API
└── shaders/              #   GLSL compute shaders (offline SPIR-V compilation)
    └── q8_0_matmul.comp  #     Batch Q8_0 matrix multiply
```

```
┌──────────────────────────────────────────────────┐
│                CLI / HTTP Layer                   │
│  calm run | calm serve | POST /v1/completions │
├──────────────────────────────────────────────────┤
│               Generation Loop                     │
│  calm_model_generate_native()                   │
│    tokenize → infer → sample → detokenize         │
│    ↕ function calling loop                        │
├─────────────────┬────────────────────────────────┤
│  Tokenizer      │  Tool Calling                   │
│  ct_tokenizer    │  ct_tools_parse, execute       │
├─────────────────┴────────────────────────────────┤
│            Transformer Inference                  │
│  ct_infer_forward → per-layer:                   │
│    RMS norm → RoPE → QKV → softmax →             │
│    attention → SwiGLU FFN → residual             │
├──────────────────────────────────────────────────┤
│           Quantized Tensor Math                   │
│  matmul_q8_0, matmul_q4_0, matmul_bq1_0,         │
│  matmul_tq1_0 (NEON SIMD on ARM)                 │
├──────────────────────────────────────────────────┤
│               GGUF Parser                         │
│  Tensor loading, metadata, tokenizer vocab,       │
│  merged_id precomputation                         │
└──────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Single-threaded inference | Model state is not thread-safe; mutex around generate() |
| Thread-per-request server | Simple, adequate for local use; no async framework |
| Zero HTTP dependencies | POSIX sockets directly; binary stays ~115KB |
| GPT-2 BPE tokenizer | Universal decoder for Qwen2.5 and LLaMA-family |
| merged_id precomputation | O(vocab×merges) → O(n) startup; ~20s → ~2s |

---

## 2. Supported Architectures

| Architecture | Inference | Verified |
|---|---|---|
| Qwen2 / Qwen2.5 | ✅ Native | ✅ Qwen2.5 0.5B Instruct |
| LLaMA 2/3 | ✅ Native | — |
| Mistral | ✅ Native | Compatible |
| Phi-3 | ✅ Native | Compatible |
| CodeLlama | ✅ Native | Compatible |
| Falcon / GPT-2 | ❌ | Not supported |
| ChatGLM / Mamba | ❌ | Not supported |

Detection: blk.{i}.attn_q.weight tensor naming (dense LLaMA-family).

---

## 3. CLI Interface (Current)

```
calm
├── run <model.gguf> [flags]    Interactive generation
│   ├── --temp FLOAT             Temperature (default: 0.0)
│   ├── --top-p FLOAT            Nucleus sampling (default: 0.95)
│   ├── --top-k INT              Top-k sampling (default: 40)
│   ├── --repeat-penalty FLOAT   Repeat penalty (default: 1.1)
│   ├── --max-tokens INT         Max tokens (default: 512)
│   └── --tools FILE             Tool definitions JSON
│
├── serve <model.gguf> [flags]   HTTP API server
│   ├── --port INT               Port (default: 8080)
│   └── --tools FILE             Tool definitions JSON
│
├── analyze <model.gguf>         Display model architecture
├── tokenize <model.gguf> <txt>  Tokenize and show token IDs
│
└── convert [flags]              Convert model format (standalone binary)
    ├── --input FILE             Source GGUF
    ├── --output FILE            Target GGUF
    └── --format STR             Target format (bq1_0, tq1_0, q4_0, q8_0)
```

---

## 4. HTTP API

### POST /v1/completions

OpenAI-compatible completions endpoint.

**Request body:**
```json
{
  "prompt": "string",
  "max_tokens": 512,
  "temperature": 0.0,
  "top_p": 0.95,
  "top_k": 40,
  "repeat_penalty": 1.1
}
```

**Response:**
```json
{
  "id": "cmpl-<hash>",
  "object": "text_completion",
  "model": "<model_path>",
  "choices": [{
    "text": "string",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 45,
    "total_tokens": 52
  }
}
```

### Tool Calling

Tools defined as JSON file, passed via `--tools`. Calm executes Qwen2.5 `<|tool_call|>` format: parses tool calls, runs built-in handlers, feeds results back to model.

**Built-in tools:**
- `get_current_time` — returns current time
- `get_weather` — mock weather for a location
- `search` — simulated web search
- `calculator` — evaluates arithmetic expressions

---

## 5. Inference Engine

### ct_infer_forward()

```
Input: hidden[n_embd]
For each layer:
  1. RMS norm → Q, K, V projections (GPU batch: 3 matmuls × 1 submit)
  2. RoPE on Q and K
  3. K_cache[pos] = K, V_cache[pos] = V
  4. Scaled dot-product attention: softmax(Q @ K_cache^T / sqrt(hd)) @ V_cache
  5. Attention output projection (GPU batch: 1 matmul)
  6. Residual connection
  7. RMS norm → SwiGLU FFN (GPU batch: gate+up 2 matmuls, then down 1 matmul)
  8. Residual connection
Final: RMS norm → output projection → logits[n_vocab]
```

### Sampling

```c
typedef struct {
    float temperature;     // 0 = greedy, >0 = scaled softmax
    float top_p;           // nucleus sampling threshold
    int top_k;             // top-k filtering
    float repeat_penalty;  // >1.0 = penalize repeated tokens
} CalmGenerateParams;
```

---

## 6. Tokenizer

GPT-2 BPE tokenizer with byte-level decoding.

- Vocab: 151,936 tokens (Qwen2.5)
- Merges: 151,387
- BOS: 151643, EOS: 151645
- Chat template: `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`
- Post-processing: Ġ → space, Ċ → newline

### Optimizations

- merged_id precomputation: O(vocab × merges) → O(length-indexed vocab groups)
- Binary search on merge pairs
- Left/right merge fields initialized to -1 for safety

---

## 7. Quantization Formats

| Format | Bits/weight | Description | Status |
|---|---|---|---|
| FP32 | 32 | Full precision | ✅ |
| Q8_0 | 9 | 8-bit with float16 scale | ✅ |
| Q4_0 | 5 | 4-bit with float16 scale | ✅ |
| TQ1_0 | ~2 | Ternary: {-1, 0, +1} | ✅ |
| BQ1_0 | ~1.125 | Binary: {-1, +1} | ✅ |

### BQ1_0 — Binary 1-bit

```
[scale: float16] [pad: 2B] [weights: 64 × 1 bit]
  uint64_t word;   // 64 weights, each bit = {-1, +1}
  float16 scale;
  pack: XOR with 0x555... to centre {0→−1, 1→+1}
```

### TQ1_0 — Ternary 1.58-bit

```
[scale: float16] [weights: 32 × 2 bit]
  uint64_t word;   // 32 weights × 2 bits
  pack: {00: −1, 10: 0, 11: +1}
```

---

## 8. Build & Dependencies

### Requirements

- C11 compiler (clang or gcc)
- POSIX system (Linux, Android/Termux, macOS, WSL)
- No external libraries — only `-lm -lpthread`

### Build

```bash
make                # calm + calm_convert
make calm           # main binary only
make calm-vk        # with Vulkan GPU backend (ARM NEON + Vulkan)
make shaders        # recompile GLSL → SPIR-V
make calm_convert   # converter only
make clean          # remove build artifacts
```

### Vulkan Requirements

- `glslangValidator` for shader compilation (or use prebuilt `shaders/q8_0_matmul_spv.h`)
- Vulkan headers (Android: included via NDK; Linux: `apt install vulkan-headers`)
- Android: `libvulkan.so` is pre-loaded at runtime (no link-time dependency)

---

## 9. Roadmap

### Done

- [x] **Vulkan GPU compute backend** (Android Adreno 730) — Q8_0 matmul offload, batch dispatch, host-coherent weights
- [x] **BQ1_0 / TQ1_0** binary/ternary quantization formats
- [x] **GGUF parser** — v3 format, any LLaMA-family architecture
- [x] **BPE tokenizer** — GPT-2 byte-level, merged-id precomputation
- [x] **HTTP API** — OpenAI-compatible `/v1/completions`
- [x] **Function calling** — Qwen2.5 `<|tool_call|>` format

### Short-term [PLANNED]

- [ ] Streaming/SSE for `/v1/completions?stream=true`
- [ ] `/v1/chat/completions` endpoint (OpenAI chat format)
- [ ] Model name in server responses (instead of filesystem path)
- [ ] Token count propagation in server response
- [ ] `--help` on subcommands

### Medium-term [PLANNED]

- [ ] Device profiler (RAM, CPU features, GPU detection)
- [ ] Model analyzer (parameter count, MoE detection, memory estimate)
- [ ] Planner / strategy engine (auto-select backend + quantization)
- [ ] MoE expert streaming (load experts on-demand from disk)
- [ ] Async I/O for server (epoll/kqueue)

### Long-term [PLANNED]

- [ ] Metal backend for Apple Silicon
- [ ] Expert cache with adaptive quantization (hot=Q4, cold=Q2/BQ1)
- [ ] Multi-model serving
- [ ] CUDA backend
- [ ] WebGPU backend (browser)

---

## 10. Performance (Current)

Measured on Snapdragon 8+ Gen 1 (ARM Calm-X2 @ 3.2 GHz), Qwen2.5 0.5B Instruct Q8_0:

| Operation | Time |
|---|---|
| Model load (with merged_id precompute) | ~2–5s |
| Prompt evaluation (7 tokens) | ~0.5s |
| Generation (per token) | ~100–200ms |
| Binary size | ~115KB |

---

## 11. Design Principles

1. **Zero external dependencies** — every line is ours; no pip, no npm, no brew
2. **Runs on anything** — phone, tablet, Raspberry Pi, laptop, server
3. **Small binary** — <200KB for full runtime
4. **Understandable** — ~8000 lines total, readable C
5. **MIT licensed** — free for any use

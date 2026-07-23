# Calm Architecture

**Версия:** 0.3
**Дата:** 23 июля 2026

---

## 1. Module Structure (Current)

```
calm                      # Main binary — CLI + generation loop
├── calm.c                #   CLI, inference orchestration, generation loop
├── calm.h                #   Public API (CalmRuntime, CalmModel, params)

├── calm_infer.c          #   Native transformer inference
│                           #   RMS norm, RoPE, SiLU, attention, SSM, SwiGLU FFN
├── calm_gguf.c           #   GGUF format parser (tensors, metadata, config)
├── calm_tokenizer.c      #   BPE tokenizer (GPT-2 byte-level, merged-id precompute)
├── calm_quant.c          #   Quantized type math (FP32, Q4_0, Q8_0, BQ1_0, TQ1_0)

├── calm_ssm.c            #   SSM (Mamba1 + Qwythos variant) forward pass
├── calm_ssm.h            #   SSM API: conv1d, selective scan, full block forward
├── calm_mla.c            #   Multi-head Latent Attention (MLA, DeepSeek-style)

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
│    │                                              │
│    ├── attention layer ──► RMS norm → RoPE → QKV │
│    │   → softmax → attention → FFN → residual    │
│    │                                              │
│    ├── SSM layer (Mamba1) ──► ssm_in → conv1d →   │
│    │   x/B/C proj → discretization →             │
│    │   selective scan → gate → out → FFN → res.  │
│    │                                              │
│    └── SSM layer (Qwythos) ──► RoPE on hidden →  │
│        fused QKV proj → SSM conv1d → alpha/beta  │
│        → selective scan → SSM out → post_attn_   │
│        norm → FFN → residual                     │
├──────────────────────────────────────────────────┤
│           Quantized Tensor Math                   │
│  matmul_q8_0, matmul_q4_0, matmul_bq1_0,         │
│  matmul_tq1_0 (NEON SIMD on ARM)                 │
│  dequant_row (BQ1_0 + TQ1_0) for conv1d          │
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
| **Qwythos (qwen35 SSM hybrid)** | **◐ SSM (NaN in forward)** | ◐ build + load OK, forward produces NaN |

Detection:
- Dense LLaMA-family: `blk.{i}.attn_q.weight` tensor naming
- SSM hybrid: `blk.{i}.ssm_in.weight` for Mamba1 layers (Jamba/Ornith)
- Qwythos variant: `blk.{i}.attn_qkv.weight` for fused QKV + `blk.{i}.ssm_conv1d.weight` per layer

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
  ─── Attention path (LLaMA-family, or every 4th layer in Jamba) ───
  1. RMS norm → Q, K, V projections (GPU batch: 3 matmuls × 1 submit)
  2. RoPE on Q and K
  3. K_cache[pos] = K, V_cache[pos] = V
  4. Scaled dot-product attention: softmax(Q @ K_cache^T / sqrt(hd)) @ V_cache
  5. Attention output projection (GPU batch: 1 matmul)
  6. Residual connection
  7. RMS norm → SwiGLU FFN (GPU batch: gate+up 2 matmuls, then down 1 matmul)
  8. Residual connection

  ─── SSM path (Mamba1 — Jamba/Ornith) ───
  1. ssm_in: input projection → 2× hidden (split for gate)
  2. ssm_conv1d: depthwise 1D convolution + bias + SiLU
  3. ssm_x: dt/B/C projections + discretization (Δ → Ā, B̄)
  4. Selective scan: h[t] = Ā·h[t-1] + B̄·x[t] (O(L), all FP32)
  5. Gate: silu(z) * y
  6. ssm_out: output projection
  7. RMS norm → SwiGLU FFN → residual

  ─── SSM path (Qwythos variant — qwen35 hybrid) ───
  1. RoPE applied directly on hidden state
  2. fused QKV: attn_qkv.weight projects hidden → Q, K, V
  3. Q, K → RoPE applied again (decoupled)
  4. ssm_conv1d: depthwise 1D conv on projected hidden (BQ1_0 dequant)
  5. ssm_alpha / ssm_beta: projections → discretization
  6. Selective scan with A, dt, B, C parameters
  7. ssm_out.weight: output projection
  8. post_attention_norm (FFN norm)
  9. SwiGLU FFN → residual
Final: RMS norm → output projection → logits[n_vocab]
```

### SSM Forward (calm_ssm.c)

Two SSM variants are supported:

**Mamba1 (`ct_forward_ssm`):** Standard Mamba1 block used by Jamba, Ornith, Qwen3.5 SSM layers.
Detected via `blk.N.ssm_in.weight`. Pipeline matches llama.cpp's `build_mamba_layer()`:
- Input projection + gate split → depthwise 1D conv + SiLU → dt/B/C projections → RMS norms (Jamba-style) → discretization → selective scan → output gate → output projection.

**Qwythos (`ct_forward_ssm_qwythos`):** Qwythos-9B-specific SSM variant (qwen35 architecture).
Detected via `blk.N.ssm_qkv.weight` — fused QKV weight in the SSM layer. Pipeline:
- Hidden → RoPE → fused QKV projection → split Q, K, V → each through independent `ssm_conv1d` → alpha/beta/A projections → discretization → selective scan (Qwythos-specific A/dt shape) → output projection → FFN norm (uses `post_attention_norm.weight` as fallback for FFN input norm).

Memory layout per SSM block:
```c
// Mamba1 state cache (per layer, per token)
float ssm_conv_state[conv_kernel-1][inner_size];  // conv1d FIFO
float ssm_h[state_size][inner_size];               // hidden state

// Qwythos state cache (per layer, per token)
float ssm_conv_state_q[3][ssm_conv_kernel-1][head_dim];  // Q conv state
float ssm_h_q[ssm_state_size][head_dim];                   // SSM hidden
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

Подробная дорожная карта → [CALM_roadmap.md](file:///data/data/com.termux/files/home/calm/CALM_roadmap.md)

### Done

- [x] **Vulkan GPU compute backend** (Android Adreno 730) — Q8_0 matmul offload, batch dispatch, host-coherent weights
- [x] **BQ1_0 / TQ1_0** binary/ternary quantization formats
- [x] **GGUF parser** — v3 format, any LLaMA-family architecture
- [x] **BPE tokenizer** — GPT-2 byte-level, merged-id precomputation
- [x] **HTTP API** — OpenAI-compatible `/v1/completions`
- [x] **Function calling** — Qwen2.5 `<|tool_call|>` format
- [x] **SSM forward pass (Mamba1)** — Jamba/Ornith/Qwen3.5 hybrid layers
- [x] **MLA forward pass** — DeepSeek-V2/V3 latent attention
- [x] **Qwythos SSM variant** — qwen35 fused-QKV SSM (builds + loads)
- [x] **DeepSeekMoE shared expert** — shared FFN over routed MoE

### Short-term [PLANNED]

- [ ] Streaming/SSE for `/v1/completions?stream=true`
- [ ] `/v1/chat/completions` endpoint (OpenAI chat format)
- [ ] Model name in server responses (instead of filesystem path)
- [ ] Token count propagation in server response
- [ ] `--help` on subcommands
- [ ] Qwythos-9B inference: debug NaN in SSM forward pass

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

# Calm — Zero-Dependency Local LLM Runtime

**Pure C, no dependencies, runs anywhere.** GGUF → tokenization → inference → tool calling → HTTP API. **Optional Vulkan GPU compute backend** for quantized matmul acceleration on mobile GPUs.

Built for ARM (phone, tablet, Raspberry Pi) and x86. Binary ~176 KB. Runs Qwen2.5 0.5B at ~5–10 tok/s on a Snapdragon 8+ Gen 1 (CPU) or **~8–15 tok/s with Vulkan GPU offload**.

**Supports SSM (Mamba1) hybrid architectures** — Jamba, Qwen3.5, Ornith, Qwythos models with selective scan layers.

## Quick Start

```bash
# Build
git clone <url>
cd calm
make

# Run inference
./calm run qwen2.5-0.5b-instruct-q8_0.gguf

# HTTP API server
./calm serve qwen2.5-0.5b-instruct-q8_0.gguf --port 8080

# With tool/function calling
./calm serve model.gguf --tools tools.json
```

## Features

- **Native inference** — RMS norm, RoPE, SiLU activation, SwiGLU FFN, multi-head attention. No llama.cpp, no Python, no CUDA.
- **SSM (Mamba1) inference** — Selective scan layers for hybrid Transformer+SSM architectures (Jamba, Qwen3.5, Ornith, Qwythos).
- **GGUF format** — Loads any LLaMA-family GGUF (Qwen2, LLaMA 2/3, Mistral, Phi-3, CodeLlama). GPT-2 BPE tokenizer with byte-level decoding.
- **HTTP API** — OpenAI-compatible `/v1/completions`. POSIX socket server, thread-per-request, zero HTTP dependencies.
- **Function/tool calling** — Qwen2.5 `<|tool_call|>` format. Built-in tools: `get_current_time`, `get_weather`, `search`, `calculator`. Custom tools via JSON config.
- **Quantized inference** — FP32, F16, Q4_0, Q4_1, Q8_0, BQ1_0 (binary), TQ1_0 (ternary) via NEON SIMD on ARM. K-quant fallback via dequant-to-F32.
- **Model conversion** — `calm_convert` converts any GGUF (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q3_K, etc.) to BQ1_0 or TQ1_0 via mmap streaming. **No full-tensor buffer** — peak RAM ≈ 1 float row (~2 MB for 9B models). Features: `--calibrate` (MSE-optimal ternary thresholds), `--verify` (post-conversion integrity check), **4-thread parallel processing**.

## Supported Architectures

| Architecture | Inference | Status |
|---|---|---|
| Qwen2 / Qwen2.5 | ✅ Native | Verified |
| LLaMA 2/3 | ✅ Native | Verified |
| Mistral | ✅ Native | Compatible |
| Phi-3 | ✅ Native | Compatible |
| CodeLlama | ✅ Native | Compatible |
| **Jamba (SSM + Attention hybrid)** | ✅ **Native SSM** | **Phase 7 — new!** |
| **Qwen3.5 (SSM hybrid)** | ✅ **Native SSM** | **Phase 7 — new!** |
| Falcon | ❌ | Not supported |
| GPT-2 / ChatGLM / Mamba2 | ❌ | Not supported |

## CLI Usage

```
calm run      <model.gguf> [prompt]          Smart launch with auto-config
calm serve    <model.gguf>                  Start HTTP API server
calm analyze  <model.gguf>                  Display model architecture info
calm scan                                   Scan device (RAM, CPU, GPU)
calm estimate <model.gguf>                  Performance prediction
calm convert  <model.gguf>                  Convert to 1-bit (BQ1_0/TQ1_0)
calm tokenize <model.gguf> <text>           Tokenize text and show tokens
calm model    info <model.gguf>             Model analysis
calm download <url>                         Download model from URL or HF

Flags for `run`:
  --backend cpu|vulkan     Force backend (default: auto-detect)
  --gpu-layers N           Layers to offload to GPU (default: 99 = all)
  --temp FLOAT             Sampling temperature (default: 0.0)
  --top-p FLOAT            Nucleus sampling (default: 0.95)
  --top-k INT              Top-k sampling (default: 40)
  --repeat-penalty FLOAT   Repeat penalty (default: 1.1)
  --max-tokens INT         Max tokens to generate (default: 20, max: 512)
  --tools FILE             Tool definitions JSON (enables function calling)

Flags for `serve`:
  --port INT               HTTP port (default: 8080)
  --tools FILE             Tool definitions JSON
```

## HTTP API

### POST /v1/completions

```json
{
  "prompt": "What is the capital of France?",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "repeat_penalty": 1.1
}
```

Response:

```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "model": "/path/to/model.gguf",
  "choices": [{
    "text": "The capital of France is Paris...",
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

Tools are defined as a JSON file and passed via `--tools tools.json`:

```json
[
  {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }
]
```

The model can call functions via `<|tool_call|>` format (Qwen2.5 Instruct), and Calm executes them and feeds results back.

## Project Structure

| File | Purpose |
|---|---|
| `calm.c` | CLI, generation loop, orchestration |
| `calm_infer.c` | Native transformer inference (attention, FFN, RoPE, SSM) |
| `calm_infer.h` | Struct definitions, API declarations |
| `calm_gguf.c` | GGUF format parser |
| `calm_tokenizer.c` | BPE tokenizer (GPT-2 byte-level) |
| `calm_quant.c` | Quantized type math (Q4_0, Q8_0, BQ1_0, TQ1_0) — NEON/AVX2 |
| `calm_quant.h` | Block struct definitions for all quant types |
| `calm_ssm.c` | SSM (Mamba1) selective scan forward pass |
| `calm_ssm.h` | SSM layer struct, function declarations |
| `calm_server.c` | HTTP API server (POSIX sockets) |
| `calm_tools.c` | Function calling engine |
| `calm_convert.c` | Model format converter (GGUF↔GGUF requantizer) |
| `calm.h` | Public API |
| `calm_vulkan.c` | Vulkan GPU compute backend (Q8_0 matmul offload) |
| `calm_vulkan.h` | Vulkan backend API |
| `shaders/q8_0_matmul.comp` | GLSL compute shader for Q8_0 batch matmul |

## Build from Source

### Requirements

- C11 compiler (clang or gcc)
- POSIX system (Linux, Android/Termux, macOS, WSL)
- No external libraries — just `-lm` for math
- **Optional**: Vulkan headers + `glslangValidator` for GPU backend

### Build

```bash
make                # Build calm + calm_convert
make calm           # Build main binary only
make calm-vk        # Build with Vulkan GPU backend (ARM NEON)
make clean          # Remove build artifacts
```

### Cross-Platform Notes

| Platform | Command |
|---|---|
| Android/Termux | `apt install clang make` → `make` |
| Linux (ARM/x86) | `apt install clang make` → `make` |
| macOS | `xcode-select --install` → `make CC=clang` |
| Windows | WSL with Ubuntu → Linux instructions |

## How It Works

1. **GGUF load** — Parses model tensors, config, tokenizer vocab
2. **Tokenizer** — GPT-2 BPE with merged-id precomputation (O(1) lookup, ~20s → ~2s startup)
3. **Inference** — Forward pass through transformer layers: RMS norm → RoPE → QKV → attention → SwiGLU FFN
4. **Decoding** — Temperature + top-p + top-k sampling with repeat penalty
5. **Output** — Text or function call via Qwen2.5 chat template

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│                  CLI / HTTP                  │
│   calm run | calm serve | POST /v1/...  │
├─────────────────────────────────────────────┤
│              Generation Loop                 │
│   tokenize → infer → sample → detokenize    │
│         ↕ function calling loop             │
├─────────────────┬───────────────────────────┤
│  Tokenizer      │  Tool Calling             │
│  (BPE, byte-dec)│  (JSON, Qwen2.5 format)   │
├─────────────────┴───────────────────────────┤
│           Transformer Inference             │
│   RMS norm → RoPE → Attn → SwiGLU → norm   │
├─────────────────────────────────────────────┤
│          Quantized Tensor Math              │
│   Q4_0 / Q8_0 / BQ1_0 / TQ1_0 via NEON     │
│   Q8_0 matmul via Vulkan compute (Adreno)   │
├─────────────────────────────────────────────┤
│               GGUF Parser                   │
│   Tensor loading, metadata, tokenizer       │
└─────────────────────────────────────────────┘
```

## Performance

Measured on Snapdragon 8+ Gen 1 (ARM Cortex-X2 @ 3.2 GHz), Qwen2.5 0.5B Instruct Q8_0:

| Operation | Time |
|---|---|
| Model load | ~2–5s (with merged_id precompute) |
| Prompt eval (7 tok) | ~0.5s |
| Generation (per token) | ~100–200ms |
| Total first token | ~0.7s |

## Limitations (MVP)

- No streaming / SSE (planned)
- No `/v1/chat/completions` — only `/v1/completions`
- Single model in memory (no swapping)
- Thread-per-request server (not async)
- Only LLaMA-family + Jamba-style hybrid architectures
- SSM backend: CPU-only (no GPU SSM kernel yet)
- Vulkan backend: Adreno-only, Q8_0 only, batch matmul only

## License

MIT — see [LICENSE](LICENSE). Use freely, modify, distribute.

## Acknowledgements

Calm was inspired by two pioneering projects:

- **[Colibri](https://github.com/JustVugg/colibri)** — Zero-dependency pure-C runtime for running 744B MoE models on consumer hardware. Showed that a self-contained C binary for LLM inference is possible and practical.

- **[Bonsai](https://prismml.com/news/bonsai-27b) by PrismML** — First 27B model fitting on iPhone via 1-bit binary quantization. Proved that extreme quantization (1.125 bits/weight) works on mobile devices. BQ1_0 and TQ1_0 formats in Calm implement similar binary/ternary packing schemes.

No code was copied from either project. Calm is an independent implementation inspired by their ideas.

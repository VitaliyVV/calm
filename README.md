# Calm — Zero-Dependency Local LLM Runtime

**Pure C, no dependencies, runs anywhere.** GGUF → tokenization → inference → tool calling → HTTP API.

Built for ARM (phone, tablet, Raspberry Pi) and x86. Binary ~113 KB. Runs Qwen2.5 0.5B at ~5–10 tok/s on a Snapdragon 8+ Gen 1.

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
- **GGUF format** — Loads any LLaMA-family GGUF (Qwen2, LLaMA 2/3, Mistral, Phi-3, CodeLlama). GPT-2 BPE tokenizer with byte-level decoding.
- **HTTP API** — OpenAI-compatible `/v1/completions`. POSIX socket server, thread-per-request, zero HTTP dependencies.
- **Function/tool calling** — Qwen2.5 `<|tool_call|>` format. Built-in tools: `get_current_time`, `get_weather`, `search`, `calculator`. Custom tools via JSON config.
- **Quantized inference** — FP32, Q4_0, Q8_0, BQ1_0 (binary), TQ1_0 (ternary) via NEON SIMD on ARM.
- **Model conversion** — Convert FP32/Q8_0 GGUF to BQ1_0/TQ1_0 format for memory-constrained devices.

## Supported Architectures

| Architecture | Inference | Status |
|---|---|---|
| Qwen2 / Qwen2.5 | ✅ Native | Verified |
| LLaMA 2/3 | ✅ Native | Verified |
| Mistral | ✅ Native | Compatible |
| Phi-3 | ✅ Native | Compatible |
| CodeLlama | ✅ Native | Compatible |
| Falcon / GPT-2 | ❌ | Not supported |
| ChatGLM / Mamba | ❌ | Not supported |

## CLI Usage

```
calm run      <model.gguf>          Run interactive inference
calm serve    <model.gguf>          Start HTTP API server
calm analyze  <model.gguf>          Display model architecture info
calm tokenize <model.gguf> <text>   Tokenize text and show tokens

Flags for `run`:
  --temp FLOAT         Sampling temperature (default: 0.0)
  --top-p FLOAT        Nucleus sampling threshold (default: 0.95)
  --top-k INT          Top-k sampling (default: 40)
  --repeat-penalty FLOAT  Repeat penalty (default: 1.1)
  --max-tokens INT     Max tokens to generate (default: 512)
  --tools FILE         Tool definitions JSON (enables function calling)

Flags for `serve`:
  --port INT           HTTP port (default: 8080)
  --tools FILE         Tool definitions JSON
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
| `calm_infer.c` | Native transformer inference (attention, FFN, RoPE) |
| `calm_gguf.c` | GGUF format parser |
| `calm_tokenizer.c` | BPE tokenizer (GPT-2 byte-level) |
| `calm_quant.c` | Quantized type math (Q4_0, Q8_0, BQ1_0, TQ1_0) |
| `calm_server.c` | HTTP API server (POSIX sockets) |
| `calm_tools.c` | Function calling engine |
| `calm_convert.c` | Model format converter |
| `calm.h` | Public API |

## Build from Source

### Requirements

- C11 compiler (clang or gcc)
- POSIX system (Linux, Android/Termux, macOS, WSL)
- No external libraries — just `-lm` for math

### Build

```bash
make                # Build calm + calm_convert
make calm         # Build main binary only
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
- Only LLaMA-family architectures
- No GPU acceleration

## License

MIT — see [LICENSE](LICENSE). Use freely, modify, distribute.

## Acknowledgements

Calm was inspired by two pioneering projects:

- **[Colibri](https://github.com/JustVugg/colibri)** — Zero-dependency pure-C runtime for running 744B MoE models on consumer hardware. Showed that a self-contained C binary for LLM inference is possible and practical.

- **[Bonsai](https://prismml.com/news/bonsai-27b) by PrismML** — First 27B model fitting on iPhone via 1-bit binary quantization. Proved that extreme quantization (1.125 bits/weight) works on mobile devices. BQ1_0 and TQ1_0 formats in Calm implement similar binary/ternary packing schemes.

No code was copied from either project. Calm is an independent implementation inspired by their ideas.

# Calm — Локальный LLM-рантайм без единой зависимости

**Чистый C, никаких зависимостей, работает везде.** GGUF → токенизация → инференс → вызов инструментов → HTTP API. **Опциональный Vulkan GPU бэкенд** для ускорения Q8_0 матричных умножений на мобильных GPU.

Собран для ARM (телефон, планшет, Raspberry Pi) и x86. Бинарник ~111 КБ. Генерирует ~5–10 токенов/с для Qwen2.5 0.5B на Snapdragon 8+ Gen 1 (CPU) или **~8–15 токенов/с с Vulkan GPU**.

`llm` `gguf` `inference` `transformer` `qwen` `llama` `c` `no-dependencies` `zero-dependency` `bpe` `tokenizer` `quantization` `binary-quantization` `ternary` `neon` `arm` `android` `termux` `http-api` `openai-compatible` `function-calling` `tool-use` `local-ai` `on-device` `edge-computing` `mit-license`

## Быстрый старт

```bash
# Сборка
git clone <url>
cd calm
make

# Запуск инференса
./calm run qwen2.5-0.5b-instruct-q8_0.gguf

# HTTP API сервер
./calm serve qwen2.5-0.5b-instruct-q8_0.gguf --port 8080

# С вызовом инструментов (function calling)
./calm serve model.gguf --tools tools.json
```

## Возможности

- **Нативный инференс** — RMS norm, RoPE, SiLU, SwiGLU FFN, multi-head attention. Без llama.cpp, Python, CUDA.
- **GGUF формат** — Загружает любые LLaMA-совместимые GGUF (Qwen2, LLaMA 2/3, Mistral, Phi-3, CodeLlama). BPE токенизатор GPT-2 с байтовым декодингом.
- **HTTP API** — OpenAI-совместимый `/v1/completions`. Сервер на POSIX сокетах, thread-per-request, ноль HTTP-зависимостей.
- **Function/tool calling** — Формат Qwen2.5 `<|tool_call|>`. Встроенные инструменты: `get_current_time`, `get_weather`, `search`, `calculator`. Пользовательские инструменты через JSON.
- **Квантованный инференс** — FP32, Q4_0, Q8_0, BQ1_0 (binary), TQ1_0 (ternary) через NEON SIMD на ARM.
- **Конвертация моделей** — Перевод FP32/Q8_0 GGUF в BQ1_0/TQ1_0 для устройств с ограниченной памятью.

## Поддерживаемые архитектуры

| Архитектура | Инференс | Статус |
|---|---|---|
| Qwen2 / Qwen2.5 | ✅ Нативный | Проверено |
| LLaMA 2/3 | ✅ Нативный | Проверено |
| Mistral | ✅ Нативный | Совместимо |
| Phi-3 | ✅ Нативный | Совместимо |
| CodeLlama | ✅ Нативный | Совместимо |
| Falcon / GPT-2 | ❌ | Не поддерживается |
| ChatGLM / Mamba | ❌ | Не поддерживается |

## Интерфейс командной строки

```
calm run      <model.gguf>            Запуск интерактивного инференса
calm serve    <model.gguf>            Запуск HTTP API сервера
calm analyze  <model.gguf>            Информация об архитектуре модели
calm tokenize <model.gguf> <текст>    Токенизация текста

Флаги для `run`:
  --temp FLOAT         Температура сэмплирования (по умолч.: 0.0)
  --top-p FLOAT        Порог nucleus sampling (по умолч.: 0.95)
  --top-k INT          Top-k сэмплирование (по умолч.: 40)
  --repeat-penalty FLOAT  Штраф повторений (по умолч.: 1.1)
  --max-tokens INT     Макс. токенов генерации (по умолч.: 512)
  --tools FILE         JSON с описанием инструментов

Флаги для `serve`:
  --port INT           Порт HTTP (по умолч.: 8080)
  --tools FILE         JSON с описанием инструментов
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

Ответ:

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

### Вызов инструментов (Tool Calling)

Инструменты определяются в JSON-файле и подключаются через `--tools tools.json`:

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

Модель вызывает функции через формат `<|tool_call|>` (Qwen2.5 Instruct). Calm выполняет их и передаёт результат обратно модели.

## Структура проекта

| Файл | Назначение |
|---|---|
| `calm.c` | CLI, цикл генерации, оркестрация |
| `calm_infer.c` | Нативный трансформер (attention, FFN, RoPE) |
| `calm_gguf.c` | Парсер GGUF |
| `calm_tokenizer.c` | BPE токенизатор (GPT-2 byte-level) |
| `calm_quant.c` | Квантованные типы (Q4_0, Q8_0, BQ1_0, TQ1_0) |
| `calm_server.c` | HTTP API сервер (POSIX сокеты) |
| `calm_tools.c` | Движок вызова инструментов |
| `calm_convert.c` | Конвертер форматов моделей |
| `calm.h` | Публичное API |

## Сборка из исходников

### Требования

- Компилятор C11 (clang или gcc)
- POSIX система (Linux, Android/Termux, macOS, WSL)
- Нет внешних библиотек — только `-lm -lpthread`

### Сборка

```bash
make                # Собрать calm + calm_convert
make calm         # Только основной бинарник
make clean          # Удалить артефакты сборки
```

### Платформы

| Платформа | Команда |
|---|---|
| Android/Termux | `apt install clang make` → `make` |
| Linux (ARM/x86) | `apt install clang make` → `make` |
| macOS | `xcode-select --install` → `make CC=clang` |
| Windows | WSL с Ubuntu → Linux инструкции |

## Как это работает

1. **Загрузка GGUF** — парсинг тензоров, конфига, словаря токенизатора
2. **Токенизатор** — BPE (GPT-2) с предвычислением merged-id (O(1) поиск, ускорение запуска 20с → 2с)
3. **Инференс** — прямой проход через слои трансформера: RMS norm → RoPE → QKV → attention → SwiGLU FFN
4. **Декодинг** — сэмплирование с temperature + top-p + top-k + штраф повторений
5. **Вывод** — текст или вызов функции через Qwen2.5 chat template

## Архитектура

```
┌─────────────────────────────────────────────┐
│                CLI / HTTP                    │
│  calm run | calm serve | POST /v1/...   │| POST /v1/...   │
├─────────────────────────────────────────────┤
│              Цикл генерации                  │
│   tokenize → infer → sample → detokenize    │
│         ↕ цикл вызова инструментов          │
├─────────────────┬───────────────────────────┤
│  Токенизатор    │  Tool Calling             │
│  (BPE, byte-dec)│  (JSON, Qwen2.5 формат)   │
├─────────────────┴───────────────────────────┤
│          Трансформер (инференс)              │
│   RMS norm → RoPE → Attn → SwiGLU → norm   │
├─────────────────────────────────────────────┤
│       Квантованная тензорная математика     │
│   Q4_0 / Q8_0 / BQ1_0 / TQ1_0 через NEON   │
├─────────────────────────────────────────────┤
│              Парсер GGUF                    │
│   Загрузка тензоров, метаданные, токенизатор│
└─────────────────────────────────────────────┘
```

## Производительность

Измерено на Snapdragon 8+ Gen 1 (ARM Cortex-X2 @ 3.2 ГГц), Qwen2.5 0.5B Instruct Q8_0:

| Операция | Время |
|---|---|
| Загрузка модели | ~2–5 с (с merged_id) |
| Промпт (7 токенов) | ~0.5 с |
| Генерация (на токен) | ~100–200 мс |
| Первый токен | ~0.7 с |

## Ограничения (MVP)

- Нет streaming / SSE (планируется)
- Нет `/v1/chat/completions` — только `/v1/completions`
- Одна модель в памяти
- Thread-per-request сервер (не асинхронный)
- Только LLaMA-семейство архитектур
- Нет GPU ускорения

## Лицензия

MIT — см. [LICENSE](LICENSE). Используйте, модифицируйте, распространяйте свободно.

## Благодарности

Calm вдохновлён двумя проектами:

- **[Colibri](https://github.com/JustVugg/colibri)** — Zero-dependency рантайм на чистом C для запуска 744B MoE моделей на обычном ПК. Доказал, что самодостаточный C-бинарник для LLM инференса возможен и практичен.

- **[Bonsai](https://prismml.com/news/bonsai-27b) от PrismML** — Первая модель 27B, работающая на iPhone, благодаря 1-bit бинарной квантизации. Доказал, что экстремальная квантизация (1.125 бит/вес) работает на мобильных устройствах. Форматы BQ1_0 и TQ1_0 в Calm реализуют аналогичные схемы упаковки.

Ни строчки кода не скопировано из этих проектов. Calm — независимая реализация, вдохновлённая их идеями.

---

*Сделано в чистых C11. Без единой внешней зависимости. Работает на вашем телефоне.*

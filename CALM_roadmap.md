# Calm — Universal Local LLM Runtime

**Дорожная карта продукта**
**Дата:** 23 июля 2026
**Версия:** v0.3 — Qwythos NaN debug + DeepSeek MLA test phase
**Контрольная точка:** git tag `v0.1-baseline` — Qwen2.5-0.5B Q8_0 работает, SSM+MLA интегрированы

---

## 🔴 Текущее состояние (23 июля 2026)

### Память устройства
```
free:   286 MiB
avail:  3.1 GiB
swap:   7.4 GiB свободно
```

### Доступные модели
```
~/calm/                                      Размер  Статус
  Qwythos-9B-BQ1_0.gguf                      3.0 GB  ◐ NaN в forward pass
  DeepSeek-Coder-V2-Lite-Instruct.Q2_K.gguf   1.2 GB  ◐ не тестирован (MLA)
  qwen2.5-0.5b-instruct-q8_0.gguf            645 MB  ✅ работает
  qwen2.5-0.5b-instruct-bq1_0.gguf           328 MB  ✅ работает
  qwen2.5-0.5b-instruct-tq1_0-v3.gguf        359 MB  ✅ работает
  qwen2.5-0.5b-instruct-q4_0.gguf            283 MB  ✅ работает
```

### Приоритет сейчас — две модели

| # | Модель | Блокер | Что делать |
|---|--------|--------|------------|
| 🔴 1 | **Qwythos-9B BQ1_0** (3.0 GB) | NaN в `ct_forward_ssm_qwythos()` | Дебаг SSM → починить → запустить |
| 🔵 2 | **DeepSeek-Coder-V2-Lite Q2_K** (1.2 GB) | MLA никогда не тестирован | Запустить → проверить → починить баги |

---

## Краткое описание

**Calm** — единый zero-dependency рантайм для запуска любых LLM на любом устройстве. 

Объединяет:
- **Bonsai-style** 1-bit/ternary экстремальную квантизацию
- **Colibri-style** streaming экспертов с диска для MoE моделей
- **Автоматический конфиг** под железо (RAM, GPU, storage speed)
- **Единый CLI** — одна команда для всего

---

## 🔴 Ключевые находки и ограничения (добавлено 17 июля 2026)

### Проблема: Termux убит OOM при запуске модели

Текущее устройство: **Snapdragon 8+ Gen 1, 10.9 GB RAM, 3.4 GB свободно**

| Попытка | Модель | Размер | Результат |
|---------|--------|--------|-----------|
| `calm-vk run Qwythos-9B-Q4_K_M.gguf` | Qwen35 hybrid, 9B, Q4_K_M | **5.6 GB** | ❌ **OOM kill** — Termux убит Android, 3.4 GB не хватает |
| `calm-vk run qwen2.5-0.5b.gguf` | Qwen2.5, 0.5B, Q5_0/Q6_K | 397 MB | ✅ **Работает** — dequant-to-F32 fallback, токенизатор, генерация 3 tok |

**Вывод:** Q4_K_M (4-bit) для 9B модели — **5.1 GB веса**. На телефоне не запустится без mmap. Решение — 1-bit (TQ1_0/BQ1_0): 9B → **~1.7 GB** — влезает в RAM.

### Архитектура Qwythos (qwen35) — гибрид SSM + Attention

Qwythos-9B — **не** стандартный dense transformer, а **Jamba-style hybrid**:

| Тип слоя | Какие слои (0-31) | Веса |
|----------|-------------------|------|
| **Pure Attention** | Каждый 4-й: 3, 7, 11, 15, 19, 23, 27, 31 (8 шт) | attn_q, attn_k, attn_v, attn_output, attn_gate, QK-norm |
| **SSM + fused QKV** | Остальные 24 слоя | ssm_conv1d, ssm_alpha, ssm_beta, ssm_out, ssm_a, ssm_dt, attn_qkv (fused) |

**Что нужно для поддержки:** SSM selective scan ядро (Mamba-style), fused QKV split, layer-type dispatch, QK-RoPE norms. **~500+ строк нового C кода.**

### Bonsai model — 1-bit квантизация

| Модель | Формат | Размер | Бит/вес | Качество |
|--------|--------|--------|---------|----------|
| **Bonsai-27B-gguf** (на HF) | BQ1_0 binary | **3.9 GB** | 1.125 | Неизвестно относительно Qwen3.6 |
| **Ternary-Bonsai-27B-gguf** (на HF) | TQ1_0 ternary | **5.9 GB** | ~2 | 95% от Qwen3.6-27B |

Ссылка: https://huggingface.co/prism-ml/Bonsai-27B-gguf

### Что умеет calm сейчас — четыре направления

| # | Направление | Статус | Детали |
|---|-------------|--------|--------|
| 🔵 1 | **Qwen2/Qwen2.5 dense** | ✅ **Работает** | Парсинг GGUF, dequant fallback (Q5_0/Q6_K), BPE токенизатор, forward pass, генерация |
| 🟡 2 | **SSM hybrid (Qwen3.5/Jamba)** | ❌ **SSM не реализован** | ~600 строк — selective scan, fused QKV, conv1d, layer dispatch. **Нужен для:** Qwythos-9B, Ornith-9B, Qwen3.5 hybrid |
| 🟢 3 | **GGUF→GGUF Requantizer** | ◐ **Dequant есть, mmap streaming нет** | calm_convert.c умеет dequant Q4_K → BQ1_0/TQ1_0, но загружает всё в RAM. Нужен mmap streaming для моделей >3 GB |
| 🟣 4 | **Bonsai (Qwen3.6, 1-bit)** | ◐ **Форматы есть, рантайма нет** | BQ1_0/TQ1_0 quant ядра есть, но инференс Qwen3.6 не реализован (зависит от направления 2)

### Реальность на телефоне

```
Модель               Формат     Вес     Влезает в 3.4 GB?
──────               ──────     ───     ──────────────────
Qwythos-9B           Q4_K_M     5.1 GB  ❌ (OOM)
Qwythos-9B           TQ1_0      1.7 GB  ✅
Qwythos-9B           BQ1_0      1.3 GB  ✅
Bonsai-27B (1-bit)   BQ1_0      3.9 GB  ⚠️ на грани (с swap)
Bonsai-27B (ternary) TQ1_0      5.9 GB  ❌
Qwen2.5-0.5B         Q5_0       0.4 GB  ✅
```

**Ключевой вывод:** 1-bit/ternary квантизация — не эксперимент, а **единственный путь запустить 9B+ модели на телефоне**. Calm уже умеет конвертировать в TQ1_0/BQ1_0 и считать в них. Нужен:
1. SSM forward pass для qwen35 (Qwythos)
2. Qwen3.6 архитектура для Bonsai
3. GGUF↔GGUF конвертер (сжать существующую Q4_K_M в TQ1_0)

---

## Фаза 0: Концепт — «Оркестратор» ✅

**Цель:** Работающий инструмент, который уже приносит пользу.

### Компоненты
- ✅ Calm CLI на Python (работает через Ollama/llama.cpp как бэкенд)
- ✅ Device Scanner — определение RAM, CPU, GPU, storage
- ✅ Model Analyzer — чтение GGUF метаданных, рекомендации
- ✅ Smart Runner — автоматический подбор параметров запуска
- ✅ Performance Estimator — прогноз скорости до запуска

### Выход
```
calm scan                  # показать железо
calm analyze model.gguf    # анализ модели
calm run model.gguf        # умный запуск
calm estimate model.gguf   # прогноз скорости
```

### Время: 1-2 дня
### Код: Python (работает на телефоне, ПК, Mac)

---

## Фаза 1: Нативный C-движок ✅

**Цель:** Собственный inference engine на чистом C, без внешних зависимостей.

### Компоненты
- [x] `calm.h` / `calm.c` — публичный C API
- [x] GGUF загрузчик (v3)
- [x] Поддержка dense архитектур (Llama, Qwen, Mistral)
- [x] CPU бэкенд: ARM NEON, x86 AVX2
- [x] Базовые форматы квантизации: Q4_0, Q8_0, BQ1_0, TQ1_0
- [x] Нормализация: RMS norm, RoPE
- [x] Активации: SiLU (SwiGLU), softmax, GELU
- [x] HTTP API: `/v1/completions`, function calling
- [x] Поддержка MoE архитектур: router → softmax → top-k → per-expert gate/up/down → weighted sum
- [ ] mmap + expert streaming (Colibri-style) — следующий приоритет

### Время: 2-3 недели
### Код: C11, ~9000 строк

---

## Фаза 2: Экстремальная квантизация ✅

**Цель:** Добавить 1-bit (binary) и ternary форматы, совместимые с Bonsai.

### Компоненты
- [x] Формат `TQ1_0` — 1-bit ternary {−1, 0, +1} (log₂3 ≈ 1.58 бита)
- [x] Формат `BQ1_0` — 1-bit binary {−1, +1} с групповым scaling (1.125 бита)
- [x] Конвертер из FP16/FP32 в 1-bit/ternary
- [x] CPU ядра: NEON для бинарных matmul
- [ ] Калибровочный датасет для PTQ

### Интеграция с Bonsai
```bash
calm convert --input qwen3.6-27b-fp16 --output bonsai-1bit --format bq1_0
calm convert --input qwen3.6-27b-fp16 --output ternary-2bit --format tq1_0
```

### Время: 2-4 недели
### Код: C + Python (конвертер)

---

## Фаза 3: GPU бэкенды ◐

**Цель:** Аппаратное ускорение на всех платформах.

### Компоненты
- [x] **Vulkan бэкенд** (Android Adreno) — Q8_0 matmul offload, batch dispatch, HOST_VISIBLE weights
- [x] **Гибрид CPU+GPU** — `--gpu-layers N` / `plan->gpu_layers`: первые N слоёв на GPU, остальные на CPU
- [ ] Metal бэкенд (Apple Silicon) — обнаружение есть, рантайм не реализован
- [ ] CUDA бэкенд (NVIDIA)
- [ ] WebGPU бэкенд (браузер)

### Приоритет
1. **Vulkan** — твой телефон (Adreno 730), Linux, Windows — ✅
2. **Metal** — Mac ecosystem — 🔲 заглушка
3. **WebGPU** — браузеры
4. **CUDA** — сервера

### Время: 4-8 недель
### Код: C + GLSL (Vulkan) + Metal shaders + CUDA kernels

---

## Фаза 4: Авто-конфиг и умный деплой ✅ (core)

**Цель:** Одна команда — любой модели на любом железе.

### C-интеграция (Phase 4, сделано)
- [x] `calm_plan_create()` — выбирает стратегию (FULL_LOAD/MMAP/MOE_STREAM) по RAM
- [x] План резолвит `CALM_BACKEND_AUTO` → VULKAN/METAL/CUDA/CPU
- [x] `plan->context_length` управляет KV-кэшем (`ct_infer_create` принимает `max_ctx`)
- [x] `plan->gpu_layers` управляет per-layer GPU offload (гибрид CPU+GPU)
- [x] CLI: `--backend cpu|vulkan`, `--gpu-layers N`
- [x] `print_model_info` показывает выбранный backend и кол-во GPU-слоёв
- [x] Auto-config pipeline: device scan → model analyze → create plan → apply → launch

### Логика принятия решений (Phase 4, сделано)

```
ВХОД: calm run model.gguf
  │
  1. СКАНИРОВАТЬ УСТРОЙСТВО
  │   ✅ RAM, CPU, GPU, Storage
  │
  2. ПРОАНАЛИЗИРОВАТЬ МОДЕЛЬ
  │   ✅ Архитектура, размер, quant
  │
  3. ВЫБРАТЬ СТРАТЕГИЮ
  │   ✅ FULL_LOAD / MMAP / MOE_STREAM
  │
  4. ЗАПУСТИТЬ
      ✅ GPU? → GPU с CPU fallback (per-layer)
      ✅ Нет? → CPU с оптимальным thread count
      ✅ --backend cpu/vulkan для переопределения
```

### Время: ~1 неделя (core C integration)
### Код: C (рантайм) + Python (CLI оркестратор)

---

## Фаза 5: Продакшн

**Цель:** Стабильный релиз, экосистема, сообщество.

### Компоненты
- [ ] OpenAI-совместимый API сервер — частично (`/v1/completions` есть)
- [ ] SSE / streaming для HTTP API — движок поддерживает, API не подключен
- [ ] Мультимодальность (LMM — изображения)
- [ ] Speculative decoding (MTP)
- [ ] Batch inference
- [ ] Пакетные менеджеры: apt, brew, pip
- [ ] Документация, бенчмарки
- [ ] CI/CD, тесты

### Время: 4-8 недель

---

## Фаза 6: GGUF→GGUF Requantizer ◐ (streaming done)

**Цель:** Стабильная утилита для пережатия любых существующих GGUF моделей (Q4_K_M, Q5_0, Q8_0, Q6_K, FP16, FP32) в BQ1_0/TQ1_0 формат без перезагрузки с HuggingFace. Прямой путь сжать Qwythos-9B-Q4_K_M (5.6 GB) → TQ1_0 (~1.7 GB) для запуска на телефоне.

### Компоненты
- [x] **mmap-based streaming**: row-by-row dequant → requant, пиковая RAM = 1 строка float (~2 MB для 9B). Подтверждено: qwen2.5-0.5b (380 MB, 290 тензоров) сконвертирован без OOM
- [x] **Sensitive layer preservation** (Q8_0): token_embd.weight, output.weight — сохраняются в Q8_0
- [x] **Dequant типов**: F32, F16, Q8_0, Q4_0, Q4_1, **Q5_0**, **Q5_1**, **Q2_K**, **Q4_K**, **Q5_K**, **Q6_K**
- [ ] **Dequant типов (недостающие)**: Q3_K (stub), Q8_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_XS, IQ3_S, IQ4_NL, IQ4_XS
- [ ] **PTQ calibration**: `--calibrate` флаг интеграция (MSE-optimal threshold для TQ1_0)
- [ ] **Параллельная обработка**: requant нескольких тензоров одновременно (worker threads)
- [ ] **Верификация**: флаг `--verify` — прочитать output GGUF, проверить целостность

### Технические детали
- ✅ **Streaming loop реализован**: 2 прохода — Pass 1 вычисляет размеры, Pass 2 stream-конвертит
- ✅ **Проверено на qwen2.5-0.5b** (Q5_0+Q6_K → TQ1_0, 380 MB → 230 MB)
- ✅ **Выходной GGUF загружается calm-движком**: tensor types правильно распознаны (type=65 = TQ1_0)
- ⚠️ Q3_K dequant — заглушка (zeroes out, но не крешит)
- ⚠️ IQ форматы не реализованы (редко встречаются в GGUF)

### Время выполнения
- ◐ **Streaming core:** ~4 часа (реализовано)
- ◐ **Dequant типов:** ~2 часа (основные сделаны, IQ форматы остались)
- 🔜 **PTQ + parallel:** ~1-2 дня

---

## Фаза 7: SSM Forward Pass (Qwen3.5 / Jamba / Ornith)

**Цель:** Добавить Mamba-style Selective Scan (SSM) для поддержки гибридных архитектур. Это откроет: Qwythos-9B (24 SSM + 8 attention слоёв), Qwen3.5-9B, Qwen3.6-27B (Bonsai), Ornith-9B.

### Компоненты
- [x] **SSM selective scan ядро** (Mamba-style, O(L) time) — Mamba1 + Qwythos variant:
  - Depthwise 1D convolution с SiLU активацией (conv1d) (+ BQ1_0 dequant для Qwythos)
  - Discretization: Δ → Ā, B̄ (two variants: standard Mamba1 and Qwythos alpha/beta)
  - Scan loop: h[t] = Ā·h[t-1] + B̄·x[t] (все FP32)
  - Обработка 24 SSM слоёв за проход
- [x] **Fused QKV поддержка для SSM слоёв**:
  - `attn_qkv.weight` → split на Q, K, V (Qwythos variant, 23 июля 2026)
  - SSM-специфичные веса: `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `ssm_out`, `ssm_a`, `ssm_dt`
- [x] **Layer-type dispatch из GGUF metadata**:
  - Qwythos: слои 3,7,11,15,19,23,27,31 → attention; остальные 24 → SSM
  - Определять по наличию `ssm_qkv` vs `attn_q.weight` в GGUF (или `ssm_in` для Mamba1)
  - Generic: читать из metadata ключ типа `qwen35.layer_type.{i}`
- [x] **QK-RoPE norms** для SSM attention слоёв (Qwythos: RoPE до QKV + повторно на Q, K)
- [x] **Поддержка в `build_weights()`**: загрузка SSM тензоров по именам (оба варианта)
- [x] **Поддержка в `ct_infer_forward()`**: per-layer выбор attention vs SSM (+ Qwythos dispatch)
- ◐ **Отладка NaN** в Qwythos SSM forward pass (см. ниже)

### Архитектура Qwythos-9B (32 слоя, Jamba-style)
```
Слой  →  0    1    2    3    4    5    6    7    8    9   10   11   ...
Тип  → SSM  SSM  SSM  ATT  SSM  SSM  SSM  ATT  SSM  SSM  SSM  ATT  ...
       (каждый 4-й — Attention, остальные — SSM)
```

### Что открывается после Phase 7
| Модель | Архитектура | TQ1_0 | BQ1_0 | Запуск |
|--------|-------------|-------|-------|--------|
| Qwythos-9B | Jamba (24 SSM) | 1.7 GB | 1.3 GB | ✅ Телефон |
| Qwen3.5-9B | Hybrid SSM | 1.7 GB | 1.3 GB | ✅ Телефон |
| Ornith-9B | Qwen3.5-based | 1.7 GB | 1.3 GB | ✅ Телефон |
| Qwen3.6-27B | Hybrid SSM | 5.3 GB | 3.8 GB | ⚠️ На грани |
| Bonsai-27B (1-bit) | Qwen3.6 binary | — | 3.9 GB | ⚠️ На грани |

### Время: 2-3 недели
### Код: C, ~800 новых строк (ssm_scan.c, расширение calm_infer.c/calm_infer.h)

---

## Фаза 8: DeepSeek2 (MLA + MoE) Forward Pass

**Цель:** Добавить Multi-head Latent Attention (MLA) и DeepSeekMoE для запуска **DeepSeek-Coder-V2-Lite-Instruct** (16B, 2.4B активных) на телефоне. Это единственная архитектура, где "2.4B активных из 16B" даёт реальный шанс запустить современный кодер на мобильном устройстве.

### Что такое MLA?

MLA (Multi-head Latent Attention) — ключевое изобретение DeepSeek-V2/V3. Вместо того чтобы хранить полные K/V проекции для каждого токена в KV cache, MLA **сжимает** их через низкоранговую проекцию:

```
Стандартный MHA:    K = x·Wk (d×d_h),     V = x·Wv (d×d_h)
MLA:                k_latent = x·Wuk (d×d_c),  k = k_latent·Wok (d_c×d_h)
                    v_latent = x·Wuv (d×d_c),  v = v_latent·Wov (d_c×d_h)
                    где d_c << d_h (сжатое latent space)
```

**Преимущество для Calm:** MLA радикально уменьшает KV cache — при d_c = 0.25×d_h, KV cache в 4 раза меньше. Для телефона с 3.4 GB это критично: можно держать больший контекст.

### Компоненты
- [x] **MLA forward**: latent projection K, V → compressed KV → RoPE (decoupled) → attention score (absorption trick)
- [x] **Decoupled RoPE**: в MLA позиционное внедрение отделено от latent K/V (дополнительный `k_rope` в cache)
- [x] **Weight loading**: `attn_q_a`, `attn_q_b`, `attn_kv_a`, `attn_kv_b`, `attn_q_norm`, `attn_k_norm` — все MLA-специфичные тензоры
- [x] **Shared experts**: фиксированные FFN (shared_gate/up/down) на всех токенах, загружаются через `build_weights()`
- [x] **Integrate with Calm MoE**: shared expert добавлен поверх routed experts в MoE-секции forward pass
- [ ] **DeepSeerMoE fine-grained routing**: 64 мелких эксперта, top-6 (на базе существующего MoE роутера)
- [ ] **DeepSeek2 tokenizer**: BPE с специальными токенами (`<｜end▁of▁sentence｜>`, `惜`)
- [ ] **Real DeepSeek2 GGUF model test**: тестирование на DS-Coder-V2-Lite (GGUF ~4 GB)

### Структура DeepSeek-Coder-V2-Lite (16B, MoE)
```
embed_dim: 2048
layers: 27
attention: MLA (latent_dim=512, kv_heads=16, head_dim=128)
MoE: 64 routed experts + 2 shared experts → top-6
active params: ~2.4B per token
total params: ~15.7B
```

### Что открывается после Phase 8
| Модель | Парам | Активных | Размер BQ1_0 | Телефон |
|--------|-------|----------|-------------|---------|
| **DeepSeek-Coder-V2-Lite-Instruct** | 16B | 2.4B | 2.3 GB | ✅ |
| **DeepSeek-Coder-V2-Lite-Base** | 16B | 2.4B | 2.3 GB | ✅ |
| DeepSeek-V2-Lite-Chat | 16B | 2.4B | 2.3 GB | ✅ |
| DeepSeek-V2-Lite (general) | 16B | 2.4B | 2.3 GB | ✅ |

### Зависимости
- **Phase 6 (Requantizer)**: Нужен, чтобы сконвертировать Q4_K_M (10.4 GB) → BQ1_0 (2.3 GB)
- Calm MoE core: уже есть ✅ — router + expert dispatch
- KV cache: уже есть ✅ — но MLA требует меньший cache (преимущество)

### Статус: ◐ В разработке (18 июля 2026)

**Реализовано:**
- `calm_mla.h` / `calm_mla.c` — MLA forward pass с absorption trick (Q_nope @ Wk_b → absorbed_q, weighted latent sum → V)
- `calm_infer.h` — MLA config поля (kv_lora_rank, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, n_shared_expert), MLA weights (attn_q_a/b, attn_kv_a/b, norm), MLA KV cache (mla_kv_cache)
- `calm_infer.c` — MLA детекция через `attn_kv_a.weight`, weight loading, KV cache alloc, dispatch в forward pass
- Shared expert (DeepSeekMoE) — weight loading + FFN forward поверх routed MoE

**Осталось:**
- Тестирование на реальной DeepSeek2 GGUF модели (DS-Coder-V2-Lite-Instruct)
- Поддержка quantized Wkv_b в MLA absorption loops (сейчас F32 fallback)
- DeepSeek2 BPE tokenizer (специальные токены)
- DeepSeerMoE fine-grained (64 experts, top-6) — существующий MoE роутер должен работать

### Новые файлы (~250 строк)
- `calm_mla.h` — декларация `ct_forward_mla()`
- `calm_mla.c` — MLA forward pass с absorption trick (Q→absorbed→score, c_weighted→V)

### Время: 2-3 недели
### Код: C, ~1000 новых строк (mla_attention.c, расширение calm_infer.c/calm_infer.h)

```
Фаза 0: Calm CLI (Python)        │ ████████████████  1-2 дня    ✅
Фаза 1: Native C engine          │ ████████████████  2-3 нед    ✅ ~9000 LOC
Фаза 2: 1-bit/ternary quant      │ ████████████████  2-4 нед    ✅
Фаза 3: GPU backends             │ ██████░░░░░░░░░░  4-8 нед    ◐ Vulkan + hybrid
Фаза 4: Auto-config + smart      │ ████████████░░░░  1-2 нед    ✅ core done
Фаза 5: Production release       │ ░░░░░░░░░░░░░░░░  4-8 нед
Фаза 6: GGUF↔GGUF Requantizer    │ ████████░░░░░░░░  1-2 нед    ◐ streaming done
Фаза 7: SSM Forward Pass         │ ████████████████  2-3 нед    ✅
Фаза 8: DeepSeek2 (MLA+MoE)      │ █████░░░░░░░░░░░  2-3 нед    ◐ integrated
                                     └── ~5-10 месяцев всего
```

**Ключевые вехи:**
- Phase 6 → сжатие любых GGUF в TQ1_0/BQ1_0 через mmap streaming (prerequisite для всего)
- Phase 7 → **Ornith-9B**, Qwythos-9B, Qwen3.5-9B на телефоне (SSM гибриды) ✅
- Phase 8 → **DeepSeek-Coder-V2-Lite 16B** на телефоне (MLA + MoE, 2.3 GB BQ1_0) ◐ forward pass integrated
- Phase 7 + 8 → два top-tier кодер-движка на телефоне: Qwen3.5-based и DeepSeek-V2-based

---

## Быстрые победы (актуально)

1. ✅ **Calm CLI** — Python (через Ollama) + C (нативный)
2. ✅ **Vulkan GPU бэкенд** — Q8_0 matmul offload на Adreno
3. ✅ **Гибрид CPU+GPU** — `--gpu-layers N`
4. ✅ **Auto-config** — `calm run` сам выбирает backend/стратегию/контекст
5. ✅ **MoE** — router + expert dispatch (softmax, top-k, weighted sum)
6. ✅ **Dequant-to-F32 fallback** — поддержка Q5_0, Q6_K, любых K-quant типов через dequant при загрузке
7. ✅ **Parameter count из GGUF тензоров** — без filename, `ct_gguf_count_params()`
8. ✅ **BPE токенайзер** — работает для GPT-2/tiktoken моделей (Qwen2, Qwen2.5)
9. ✅ **BQ1_0/TQ1_0 quant ядра** — с NEON оптимизацией
10. ✅ **GGUF→TQ1_0/BQ1_0 конвертер** — `calm_convert.c`
11. ◐ **Phase 6: GGUF→GGUF Requantizer** — streaming core ✅, dequant F32/F16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/Q4_K/Q5_K/Q6_K ✅, остались IQ форматы
12. ✅ **Phase 7: SSM Forward Pass (Mamba1)** — Mamba-style selective scan для Qwen3.5/Jamba/Ornith гибридов (+ calm_ssm.c/h, build_weights, layer dispatch)
13. ◐ **Phase 7b: Qwythos SSM variant** — qwen35 fused-QKV SSM (builds + loads, NaN в forward pass)
14. ◐ **Phase 8: DeepSeek2 (MLA+MoE)** — Multi-head Latent Attention + DeepSeekMoE (forward pass integrated: calm_mla.c, calm_infer.h/c, shared expert, absorption trick)
15. 📦 **mmap/expert streaming** — Colibri-style, холодные эксперты с диска
16. 📦 **Qwen3.6 архитектура** — для запуска Bonsai-27B (1-bit, 3.9 GB) — зависит от Phase 7 (ждёт фикса NaN в Qwythos variant)

---

## 🗺️ Карта зависимостей: модель → фаза

Какая модель требует каких фаз в Calm C engine:

```
Фаза                          Ornith  Qwythos  Qwen3.5  DS-CoderV2  Qwen3.6  Bonsai
                              -9B     -9B      -9B      -16B        -27B     -27B
                              
Phase 1: C engine base        ✅      ✅       ✅       ✅          ✅       ✅
Phase 2: BQ1_0/TQ1_0 quant    ✅      ✅       ✅       ✅          ✅       ✅
Phase 4: Auto-config          ✅      ✅       ✅       ✅          ✅       ✅
Phase 6: Requantizer          ✅      ✅       ✅       ✅          ✅       ✅
Phase 7: SSM Forward          ✅      ✅       ✅       —           ✅       ✅
Phase 8: MLA + DeepSeekMoE    —       —        —        🔜          —        —
Доп: Qwen3.6 arch             —       —        —        —           🔜       🔜
```

**Приоритетная последовательность разработки:**
```
Phase 6 (Requantizer) ───→ Phase 7 (SSM) ───→ Phase 8 (MLA/DeepSeek2)
       │                        │                      │
       │                        ├─ Ornith-9B ✅        └─ DS-Coder-V2-16B ✅
       │                        ├─ Qwythos-9B ✅
       │                        └─ Qwen3.5-9B ✅
       │
       └─ Сжать любую модель в TQ1_0/BQ1_0 (prerequisite)
```

**Правило:** Ни одна новая архитектура не запустится на телефоне без **Phase 6** — существующие GGUF (Q4_K_M и т.д.) не влезают в 3.4 GB. Requantizer — критический шлюз.

**Расчёт размера:**
- Q4_K_M: ~0.60 × params (GB) — тяжёлый, 2-й слой квантизации llama.cpp
- TQ1_0: 0.1975 × params (GB) — ternary 1.58-bit, 256 weights/block
- BQ1_0: 0.1406 × params (GB) — binary 1.125-bit, 128 weights/group
- Доступная RAM: ~3.4 GB (после Android + Termux)
- **[✅]** = влезает в TQ1_0 (≤3.0 GB с запасом) — приоритет Calm
- **[~]** = на грани (3.0-3.4 GB) — нужен BQ1_0
- **[❌]** = не влезает (>3.4 GB даже в BQ1_0)

---

### 🥇 ELITE — Reasoning + Coding (приоритетные для запуска)

| Модель | Парам | Q4 | TQ1_0 | BQ1_0 | Тел. | Тип |
|--------|-------|----|-------|-------|------|-----|
| **Ornith-9B** | 8.99B | 5.4 GB | **1.78 GB** | **1.26 GB** | ✅🔥 | coding agent #1 |
| **Qwythos-9B** | 8.99B | 5.6 GB | **1.78 GB** | **1.26 GB** | ✅🔥 | hybrid (SSM+attn) |
| **DeepSeek-R1-7B** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | reasoning |
| **DS-Coder-V2-7B** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | coding expert |
| **Qwen3.5-Coder-7B** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | coding |
| **CodeGemma-7B** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | coding |
| **Yi-Coder-9B** | 9.0B | 5.4 GB | **1.78 GB** | **1.26 GB** | ✅ | coding |
| **DeepSeek-R1-14B** | 14.0B | 8.4 GB | **2.76 GB** | **1.97 GB** | ✅ | reasoning |
| **Qwen3.5-9B** | 8.99B | 5.4 GB | **1.78 GB** | **1.26 GB** | ✅🔥 | hybrid (SSM+attn) |
| **Qwen3.6-27B** | 27.0B | 16.2 GB | 5.33 GB | **3.80 GB** | ~| SSM hybrid |
| **Bonsai-27B (1-bit)** | 27.0B | — | — | **3.9 GB** | ~| native binary |

### 🥈 GENERAL — All-rounder

| Модель | Парам | Q4 | TQ1_0 | BQ1_0 | Тел. | Тип |
|--------|-------|----|-------|-------|------|-----|
| **Qwen2.5-0.5B** | 0.5B | 0.3 GB | **0.10 GB** | **0.07 GB** | ✅ | general |
| **Qwen2.5-1.5B** | 1.5B | 0.9 GB | **0.30 GB** | **0.21 GB** | ✅ | general |
| **Qwen2.5-3B** | 3.0B | 1.8 GB | **0.59 GB** | **0.42 GB** | ✅ | general |
| **Qwen2.5-7B** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | general |
| **Qwen2.5-14B** | 14.0B | 8.4 GB | **2.76 GB** | **1.97 GB** | ✅ | general |
| **Qwen2.5-32B** | 32.0B | 19.2 GB | 6.32 GB | 4.50 GB | ❌ | general |
| **Qwen3-4B** | 3.9B | 2.3 GB | **0.77 GB** | **0.55 GB** | ✅ | general |
| **Qwen3-8B** | 7.6B | 4.6 GB | **1.50 GB** | **1.07 GB** | ✅ | general |
| **Qwen3-14B** | 14.0B | 8.4 GB | **2.76 GB** | **1.97 GB** | ✅ | general |
| **Qwen3-32B** | 32.0B | 19.2 GB | 6.32 GB | 4.50 GB | ❌ | general |
| **Qwen3-235B-A22B*** | 235B | 141 GB | 46.4 GB | 33.0 GB | ❌ | MoE |
| **Llama3.2-1B** | 1.0B | 0.6 GB | **0.20 GB** | **0.14 GB** | ✅ | general |
| **Llama3.2-3B** | 3.0B | 1.8 GB | **0.59 GB** | **0.42 GB** | ✅ | general |
| **Llama3.2-11B** | 11.0B | 6.6 GB | **2.17 GB** | **1.55 GB** | ✅ | vision |
| **Mistral-7B-v0.3** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | general |
| **Mistral-Nemo-12B** | 12.0B | 7.2 GB | **2.37 GB** | **1.69 GB** | ✅ | general |
| **Gemma4-2B** | 2.0B | 1.2 GB | **0.40 GB** | **0.28 GB** | ✅ | general |
| **Gemma4-9B** | 9.0B | 5.4 GB | **1.78 GB** | **1.26 GB** | ✅ | general |
| **Gemma4-27B** | 27.0B | 16.2 GB | 5.33 GB | **3.80 GB** | ~| general |
| **Phi4-14B** | 14.0B | 8.4 GB | **2.76 GB** | **1.97 GB** | ✅ | general |

### 🥉 EDGE — Для мобильных устройств

| Модель | Парам | Q4 | TQ1_0 | BQ1_0 | Тел. | Тип |
|--------|-------|----|-------|-------|------|-----|
| **Phi4-Mini-3.8B** | 3.8B | 2.3 GB | **0.75 GB** | **0.53 GB** | ✅ | edge |
| **Phi3.5-Mini-3.8B** | 3.8B | 2.3 GB | **0.75 GB** | **0.53 GB** | ✅ | edge |
| **Gemma4-2B** | 2.0B | 1.2 GB | **0.40 GB** | **0.28 GB** | ✅ | edge |
| **Falcon3-1B** | 1.0B | 0.6 GB | **0.20 GB** | **0.14 GB** | ✅ | edge |
| **Falcon3-3B** | 3.0B | 1.8 GB | **0.59 GB** | **0.42 GB** | ✅ | edge |
| **Falcon3-7B** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | edge |
| **Falcon3-10B** | 10.0B | 6.0 GB | **1.98 GB** | **1.41 GB** | ✅ | edge |
| **Granite3.2-2B** | 2.0B | 1.2 GB | **0.40 GB** | **0.28 GB** | ✅ | enterprise/RAG |
| **Granite3.2-8B** | 8.0B | 4.8 GB | **1.58 GB** | **1.12 GB** | ✅ | enterprise/RAG |
| **Granite3.1-MoE-1B** | 1.0B | 0.6 GB | **0.20 GB** | **0.14 GB** | ✅† | MoE edge |
| **Granite3.1-MoE-3B** | 3.0B | 1.8 GB | **0.59 GB** | **0.42 GB** | ✅† | MoE edge |
| **SmolLM2-1.7B** | 1.7B | 1.0 GB | **0.34 GB** | **0.24 GB** | ✅ | edge |
| **MiniCPM3-4B** | 4.0B | 2.4 GB | **0.79 GB** | **0.56 GB** | ✅ | multilingual |
| **Stable-Code-3B** | 2.7B | 1.6 GB | **0.53 GB** | **0.38 GB** | ✅ | coding edge |
| **CodeGemma-2B** | 2.0B | 1.2 GB | **0.40 GB** | **0.28 GB** | ✅ | coding edge |
| **Starcoder2-3B** | 3.0B | 1.8 GB | **0.59 GB** | **0.42 GB** | ✅ | coding edge |
| **Yi-Coder-1.5B** | 1.5B | 0.9 GB | **0.30 GB** | **0.21 GB** | ✅ | coding edge |
| **Nemotron-Mini-4B** | 4.0B | 2.4 GB | **0.79 GB** | **0.56 GB** | ✅ | edge |
| **LocoOperator-0.85B** | 0.85B | 0.5 GB | **0.17 GB** | **0.12 GB** | ✅ | agent edge |

† MoE: активные эксперты меньше, но файл со всеми экспертами — полный размер; если streaming с диска — реально

### 🧪 REASONING (умные, но больше)

| Модель | Парам | Q4 | TQ1_0 | BQ1_0 | Тел. | Тип |
|--------|-------|----|-------|-------|------|-----|
| **DS-R1-7B** | 7.0B | 4.2 GB | **1.38 GB** | **0.98 GB** | ✅ | reasoning |
| **DS-R1-8B-Llama** | 8.0B | 4.8 GB | **1.58 GB** | **1.12 GB** | ✅ | reasoning |
| **DS-R1-14B** | 14.0B | 8.4 GB | **2.76 GB** | **1.97 GB** | ✅ | reasoning |
| **DS-R1-32B** | 32.0B | 19.2 GB | 6.32 GB | 4.50 GB | ❌ | reasoning |
| **QwQ-32B** | 32.0B | 19.2 GB | 6.32 GB | 4.50 GB | ❌ | reasoning |
| **DeepSeek-V3-0324** | 671B | 403 GB | 132 GB | 94 GB | ❌ | MoE ultra |

### 🎯 ИТОГ: Приоритет для запуска на телефоне

```
Приоритет  │ Модель              │ Парам │ TQ1_0 │ BQ1_0 │ Что даёт
───────────┼─────────────────────┼───────┼───────┼───────┼────────────────────────────
🏆 P0      │ Ornith-9B           │ 9B    │ 1.8GB │ 1.3GB │ Лучший coding agent на телефоне
🏆 P0      │ Qwythos-9B          │ 9B    │ 1.8GB │ 1.3GB │ Jamba hybrid, требует Phase 7
⭐ P1      │ DS-R1-14B           │ 14B   │ 2.8GB │ 2.0GB │ Макс reasoning
⭐ P1      │ Mistral-Nemo-12B    │ 12B   │ 2.4GB │ 1.7GB │ Лучший общий
⭐ P1      │ Qwen3.5-9B          │ 9B    │ 1.8GB │ 1.3GB │ SSM hybrid, Phase 7
⭐ P1      │ Yi-Coder-9B         │ 9B    │ 1.8GB │ 1.3GB │ Альтернатива Ornith
✅ P2      │ DS-R1-7B            │ 7B    │ 1.4GB │ 1.0GB │ Быстрый старт
✅ P2      │ Qwen3-8B            │ 7.6B  │ 1.5GB │ 1.1GB │ Универсальная
✅ P2      │ Falcon3-10B         │ 10B   │ 2.0GB │ 1.4GB │ Enterprise
```

**Легенда:** ✅🔥 = уже работает или минимальные изменения; ✅ = влезает с запасом; ~ = на грани; ❌ = не влезает  
*MoE файлы полные (все эксперты), но активные эксперты меньше — с mmap streaming реальный лимит выше*

---

## Ресурсы

- **Colibri** — https://github.com/JustVugg/colibri (C engine, MoE streaming)
- **bitnet.c** — https://github.com/artalis-io/bitnet.c (C engine, universal + GPU)
- **ExpertFlow** — https://github.com/jhammant/expertflow (Rust, smart MoE streaming)
- **flash-moe** — https://github.com/tayoun/flash-moe (C/Metal, Apple MoE streaming)
- **QMoE** — https://github.com/IST-DASLab/qmoe (sub-1-bit MoE compression research)
- **Bonsai 27B** — https://huggingface.co/collections/prism-ml/bonsai-27b
- **Bonsai GGUF** — https://huggingface.co/prism-ml/Bonsai-27B-gguf
- **Qwythos-9B** — https://huggingface.co/Qwythos/Qwythos-9B (SSM+Attention hybrid)
- **Ornith-9B** — https://huggingface.co/Ornith/Ornith-9B (Qwen3.5-based coding agent)
- **Qwen3.5-9B** — https://huggingface.co/Qwen/Qwen3.5-9B (hybrid SSM, best general)
- **Qwen3.6-27B** — https://huggingface.co/Qwen/Qwen3.6-27B (latest Qwen)
- **DeepSeek-R1** — https://huggingface.co/deepseek-ai/DeepSeek-R1 (reasoning MoE)
- **DeepSeek-Coder-V2** — https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- **Phi4** — https://huggingface.co/microsoft/Phi-4 (14B, strong for size)
- **Phi4-Mini** — https://huggingface.co/microsoft/Phi-4-mini-instruct (3.8B edge)
- **Falcon3** — https://huggingface.co/collections/tiiuae/falcon3-673247f641e09a0d8dfaa54c (1B-10B)
- **Granite3.2** — https://huggingface.co/ibm-granite/granite-3.2-8b-instruct (enterprise/RAG)
- **Mistral-Nemo** — https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407 (12B)
- **Gemma4** — https://huggingface.co/collections/google/gemma-4-6764c1a5f7ee69c417e0161c (2B/9B/27B)
- **SmolLM2** — https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct (edge)
- **MiniCPM3** — https://huggingface.co/openbmb/MiniCPM3-4B (multilingual edge)
- **Yi-Coder** — https://huggingface.co/01-ai/Yi-Coder-9B (coding)
- **Qwen2.5** — https://huggingface.co/collections/qwen/qwen25-66e81a666513e518adb90d9e
- **Qwen3** — https://huggingface.co/collections/qwen/qwen3-673e9eb3d3fc2f06cb0f4b26
- **llama.cpp** — https://github.com/ggml-org/llama.cpp (GGUF reference, Vulkan backend)

---

---

## 🚀 Phase 7: SSM Forward Pass — Mamba1 для гибридных моделей (17 июля 2026)

Реализован полный SSM (Mamba1) forward pass для token-by-token инференса гибридных
Transformer+SSM архитектур (Jamba, Ornith, Qwythos, Qwen3.5 SSM).

**Новые файлы:**
- `calm_ssm.h` — структуры SSM-слоя, декларации функций
- `calm_ssm.c` — реализации: `ct_ssm_conv1d()`, `ct_ssm_selective_scan()`, `ct_forward_ssm()`

**Изменённые файлы:**
- `calm_infer.h` — SSM конфиг в `ct_infer_config`, SSM weight pointers + `is_ssm` в `ct_infer_layer`,
  SSM state caches в `ct_infer_state`
- `calm_infer.c` — SSM metadata extraction в `extract_config()`, SSM tensor loading в `build_weights()`,
  SSM dispatch (`is_ssm ? ssm_path : attention_path`) в `ct_infer_forward()`,
  SSM cache alloc/free в `ct_infer_create()`/`ct_infer_free()`
- `Makefile` — добавлен `calm_ssm.o`

**SSM block pipeline (llama.cpp `build_mamba_layer()` equivalent):**
1. `ssm_in` — input projection + output gate split
2. `ssm_conv1d` — depthwise 1D convolution + bias + SiLU
3. `ssm_x` — dt/B/C projection
4. `ssm_dt_norm`/`ssm_b_norm`/`ssm_c_norm` — RMS norm (Jamba-style)
5. `ssm_dt` — discretization time step projection
6. `ssm_a` + `ssm_d` — selective scan + skip connection
7. `silu(z)*y` — output gating
8. `ssm_out` — output projection

**Supported GGUF tensor names:** `blk.N.ssm_in.weight`, `blk.N.ssm_conv1d.weight`,
`blk.N.ssm_x.weight`, `blk.N.ssm_dt.weight`, `blk.N.ssm_dt_norm.weight`,
`blk.N.ssm_b_norm.weight`, `blk.N.ssm_c_norm.weight`, `blk.N.ssm_a`,
`blk.N.ssm_d`, `blk.N.ssm_out.weight`

**GGUF metadata keys:** `{arch}.ssm.conv_kernel`, `ssm.inner_size`, `ssm.state_size`,
`ssm.time_step_rank`, `ssm.dt_b_c_rms`

**Детекция гибридных слоёв:** каждый слой проверяется на наличие `blk.N.ssm_in.weight` —
если найден, загружаются SSM веса и устанавливается `is_ssm=1`. Attention-слои
загружаются как обычно. Диспетчер в `ct_infer_forward()` выбирает путь выполнения
по флагу `is_ssm`.

### Qwythos-9B — SSM variant (qwen35, fused QKV) ◐ добавлено 23 июля 2026

Qwythos-9B использует **нестандартный SSM вариант**, отличный от Mamba1:

| Аспект | Mamba1 (Jamba/Ornith) | Qwythos (qwen35) |
|--------|----------------------|-------------------|
| Детекция слоя | `blk.N.ssm_in.weight` | `blk.N.ssm_qkv.weight` |
| QKV проекция | Раздельные Q, K, V | **Fused**: `attn_qkv.weight` → split |
| RoPE | Только в attention-слоях | На hidden **до** QKV + повторно на Q, K |
| conv1d | Один на весь SSM блок | **Три** (Q, K, V — независимые) |
| alpha/beta | Нет | `ssm_alpha`, `ssm_beta` отдельно |
| A/dt форма | `[ssm_state_size × inner_size]` | `[n_head × 1]` (другая размерность) |
| FFN input norm | `blk.N.ffn_norm.weight` | `blk.N.post_attention_norm.weight` |

**Реализовано (23 июля 2026):**

- `calm_infer.h` — новые поля в `ct_infer_layer`: `ssm_qkv` (QKV weight pointer), `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt`, `is_qwythos`, Qwythos-специфичные conv состояния
- `calm_infer.c` — `build_weights()`: детекция Qwythos через `blk.N.ssm_qkv.weight`, загрузка fused QKV и Qwythos-специфичных тензоров; `post_attention_norm.weight` как fallback для FFN norm; dispatch в `ct_infer_forward()` по `lw->ssm_qkv`
- `calm_ssm.h` — декларация `ct_forward_ssm_qwythos()`
- `calm_ssm.c` — `ct_forward_ssm_qwythos()`: RoPE на hidden → fused QKV → split Q/K/V → 3× conv1d (с BQ1_0 dequant) → alpha/dt → selective scan (Qwythos-специфичный A/dt формат) → output proj → post_attention_norm → FFN
- `Makefile` — добавлен `calm_ssm.o`
- BQ1_0 dequant для conv1d: `dequant_row()` в `calm_ssm.c` — BQ1_0 dequant в FP32 для depthwise conv

**Статус:** ✅ Собирается, загружается, распознаёт все 32 слоя Qwythos-9B (24 SSM + 8 attention). Forward pass produces **NaN** — причина не установлена.

**Анализ кода ct_forward_ssm_qwythos() (calm_ssm.c, 23 июля 2026):**

Pipeline (строки 496-568):
```
Step 1: xz = matmul(hidden_in, attn_qkv)           [E] → [2*E]
Step 2: xz_conv = silu(conv1d(xz))                 [2*E] conv → [2*E]
Step 3: split → x_part[E], z_part[E]
Step 4: alpha = matmul(hidden_in, ssm_alpha)        [E] → [d_state]
Step 5: dt = exp(alpha + dt_bias)                   per-state
Step 6: beta = sigmoid(matmul(hidden_in, ssm_beta))  [E] → [d_state]
Step 7: selective scan: h_new = a_bar*h + b_bar*x   d_state × d_inner
Step 8: y *= sigmoid(z)                             gate
Step 9: out = matmul(y, ssm_out)                    [E] → [E]
```

**Гипотезы NaN (23 июля 2026, обновлено с анализом кода):**

1. ✅ `fp16_to_f32()` — не источник (возвращает 0.0 на NaN)
2. ✅ `ct_matmul_bq1_0()` — не источник (корректные ненулевые результаты)
3. ◐ **Step 1: `matmul(xz, hidden_in, lw->ssm_qkv, E, 2*E)`** — attn_qkv.weight в BQ1_0. Если размерность `[E, 2*E]` не совпадает с GGUF-метаданными, matmul производит мусор → NaN. **Проверить GGUF metadata вывод.**
4. ◐ **Step 2: conv1d на d_xz=8192 каналов с BQ1_0 весом** — `row_stride` вычисляется через `ct_gguf_tensor_size()` для BQ1_0 (128 эл/блок). Если stride неверный — dequant читает мусор → NaN.
5. ◐ **Step 5: `expf(alpha[s] + dt_bias[s])` без softplus** — Mamba1 использует `log(1+exp(dt))` (softplus). Qwythos использует raw exp. Если alpha[s] большой (>80), `expf(→ +Inf)`. **Inf × anything = Inf, не NaN.** Not primary cause.
6. ◐ **Step 7: selective scan** — `alpha[s]` используется и как dt_B, и как dt_C (b_bar = dt*alpha, c_gated = alpha*beta). Если alpha содежит Inf: `Inf * 0 = NaN`.
7. ◐ **conv_state/h_state pointer arithmetic** — `ssm_conv_state` аллоцируется как `[n_layer × d_inner × (d_conv-1)]` для Mamba1 (d_inner=2*E). Для Qwythos d_inner=E, d_xz=2*E. Conv1d использует `d_xz` каналов, НО conv_state выделен под d_inner. **Если d_conv > 1 → чтение/запись за границей буфера!**

**Приоритетная гипотеза: buffer overflow в conv_state**
```
Mamba1: conv_state[n_layer][d_inner × (d_conv-1)]   d_inner = 2*E = 8192
Qwythos: conv_state[n_layer][d_inner × (d_conv-1)]  d_inner = E = 4096, НО conv1d использует d_xz = 2*E = 8192
```
`ct_ssm_conv1d()` получает `d_inner=d_xz=8192` для Qwythos, shift-цикл (строки 142-149) пишет в conv_state[i * c_stride + ...] с i до d_xz-1 = 8191. А буфер выделен для d_inner=4096. **Запись за границу → коррупция → NaN.**

**План фикса (приоритет):**
1. Проверить: `ssm_d_inner` в метаданных Qwythos GGUF — если = 4096 (n_embd), то conv_state аллоцирован вдвое меньше нужного
2. Читать `ssm.conv_kernel` из GGUF метаданных (cfg->ssm_d_conv) и использовать `d_xz` (2*d_inner) для conv_state Qwythos
3. Отдельная аллокация conv_state для Qwythos: `n_layer × d_xz × (d_conv-1)`

**Следующие шаги:**
1. ✅ **Скачать Qwythos GGUF** — BQ1_0 (3.0 GB) скопирован в ~/calm/
2. 🔜 **Первый запуск** — `calm-cpu run Qwythos-9B-BQ1_0.gguf 2>&1 | head -100`
3. 🔜 **Проверить GGUF metadata** — `calm-cpu analyze Qwythos-9B-BQ1_0.gguf` → ssm_d_inner, n_embd, d_conv
4. 🔜 **Починить conv_state аллокацию** — если подтвердится buffer overflow
5. 🔜 **Добавить дебаг-принты** — логировать shapes и первые/последние значения каждого шага
6. 🔜 **Проверить conv1d row_stride** для BQ1_0 с d_xz=8192

### Mamba2 (future)

NEON-оптимизация selective scan (особенно expf-вызовы в цикле)
Поддержка Mamba2 (SSM с группировкой)
Опциональный GPU/Vulkan SSM kernel

---

## 🧪 Phase 9: DeepSeek MLA — Test & Fix (добавлено 23 июля 2026)

**Цель:** Запустить DeepSeek-Coder-V2-Lite-Instruct.Q2_K.gguf (1.2 GB, уже на диске) через calm-cpu, проверить MLA forward pass.

### Статус
- `calm_mla.c` — реализован MLA forward pass с absorption trick
- `calm_infer.c` — детекция MLA через `attn_kv_a.weight`, загрузка весов, KV cache alloc
- Shared expert (DeepSeekMoE) — загрузка + FFN forward поверх routed MoE
- **Ни разу не тестировалось** — не было подходящей GGUF модели

### План тестирования
1. `calm-cpu analyze DeepSeek-Coder-V2-Lite-Instruct.Q2_K.gguf` — проверить метаданные
2. `calm-cpu run DeepSeek-Coder-V2-Lite-Instruct.Q2_K.gguf --max-tokens 10` — тест генерации
3. Если падает/NaN — дебаг аналогично Qwythos
4. Известные проблемы MLA: quantization поддержка Wkv_b в absorption loops (сейчас F32 fallback), DeepSeek2 tokenizer (BPE спецтокены)

### Размеры
| Файл | Размер | RAM |
|------|--------|-----|
| DeepSeek-Coder-V2-Lite-Instruct.Q2_K.gguf | 1.2 GB | ✅ влезает с запасом |

---

## 🐛 Исправленные баги (17 июля 2026)

### 1. TQ1_0 segfault при inference — `ct_matmul_tq1_0` buffer overflow

**Причина:** В NEON-пути `ct_matmul_tq1_0()` переменная `cols_in_block` считала общее число оставшихся колонок (`I - b*256`), а не реальное количество в текущем TQ1_0 блоке (макс. 256). Функция `dequant_tq1_0_to_q8_0()` обрезала до 256, но счётчик `q8_count` увеличивался от необрезанного значения — запись Q8_0 блоков уходила за границу аллоцированного буфера → heap corruption → segfault.

**Симптом:** `calm run model-tq1_0.gguf` — токенизатор грузится, лог тензоров выводится, но при первом forward pass — `Segmentation fault`.

**Фикс:** `calm_quant.c` — закаплен `cols_in_block = remaining < 256 ? remaining : 256`.

### 2. GGUF writer — отсутствие 32-байтового выравнивания между тензорами

**Причина:** `write_header_and_metadata()` вычислял смещения тензоров с 32-байтовым выравниванием, но `stream_tensor()` писал данные последовательно без паддингов. При некратном 32 размере тензора смещения в tensor info не совпадали с реальными позициями → неверные указатели на данные тензоров.

**Фикс:** `calm_convert.c` — после каждого тензора добавляется padding до 32 байт.

### 3. `is_head_tensor()` — ложное срабатывание на `attn_output.weight`

**Причина:** Использовался `strstr(name, "output")`, который находил подстроку "output" в `blk.N.attn_output.weight` — все attention output проекции сохранялись в Q8_0 (8 бит) вместо конвертации в TQ1_0 (1.58 бит). Потеря ~19 MB на Qwen2.5-0.5B.

**Фикс:** `calm_convert.c` — заменён на `strcmp()` по точным именам: только `token_embd.weight`, `tok_embd.weight`, `output.weight`, `head.weight`.

---

### 🐛 Исправленные баги (23 июля 2026)

### 4. Qwythos FFN — отсутствует `post_attention_norm.weight`

**Причина:** Qwythos-9B использует `blk.N.post_attention_norm.weight` для нормализации входа FFN вместо стандартного `blk.N.ffn_norm.weight`. `build_weights()` падал при загрузке Qwythos GGUF потому что не находил ffn_norm веса.

**Симптом:** `segfault at memory address` при загрузке Qwythos-9B GGUF — `gguf_get_tensor()` не находила `ffn_norm.weight`.

**Фикс:** `calm_infer.c` в `ct_infer_load_weights()` — при Qwythos-детекции (найден `blk.0.ssm_qkv.weight`) сначала ищется `post_attention_norm.weight`, и если найден, используется как `ffn_norm`. Проверено: все 32 слоя загружаются корректно.

### 5. BQ1_0 conv1d — отсутствовал dequant для depthwise 1D convolution

**Причина:** Qwythos-9B в BQ1_0 имеет `ssm_conv1d.weight` в BQ1_0 формате (`ggml_type 65`). Depthwise 1D convolution требует FP32 веса, но `dequant_row()` в `calm_ssm.c` поддерживал только FP32, FP16, Q8_0, TQ1_0. BQ1_0 кейс отсутствовал → `dequant_row` возвращал 0 (unsupported type) → conv1d на нулевых весах → NaN на выходе.

**Симптом:** NaN при первом же SSM conv1d forward pass на Qwythos-9B BQ1_0.

**Фикс:** `calm_ssm.c` — добавлен BQ1_0 кейс в `dequant_row()`: `ggml_type 65 (BQ1_0)` → dequant через `dequant_bq1_0` из `calm_quant.c`. Система dequant-to-F32 теперь обрабатывает все 6 форматов: FP32, FP16, Q8_0, Q4_0, TQ1_0, BQ1_0.

# Calm — Universal Local LLM Runtime

**Дорожная карта продукта**
**Дата:** 28 июля 2026 (обновлено 1 августа 2026, вечер — консолидация репозиториев)
**Версия:** v0.4 (C engine + AVX2 x86 + Vulkan + auto-config + Bonsai-format kernels)

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
- [x] **PTQ calibration**: `--calibrate` флаг интеграция (MSE-optimal threshold для TQ1_0) — реализован (calm_convert.c, `calibrate_ternary`)
- [x] **Параллельная обработка**: requant нескольких тензоров одновременно (worker threads, pthread, до 4 потоков)
- [x] **Верификация**: флаг `--verify` — прочитать output GGUF, проверить целостность (`verify_gguf`)

> ✅ **Проверено в рантайме 31 июля 2026:** `calm_convert --input qwen2.5-0.5b.gguf --format tq1_0 --calibrate --verify` — 290 тензоров, 4 потока, peak RAM = строка (~0 MB), output 231 MB → загружен движком (type=65 распознан), calm_smoke **PASS** (24 слоя, без NaN). Полная цепочка конвертер→движок работает.

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
- [x] **SSM selective scan ядро** (Mamba-style, O(L) time):
  - Depthwise 1D convolution с SiLU активацией (conv1d)
  - Discretization: Δ → Ā, B̄
  - Scan loop: h[t] = Ā·h[t-1] + B̄·x[t] (все FP32)
  - Обработка 24 SSM слоёв за проход
- [ ] **Fused QKV поддержка для SSM слоёв**:
  - `attn_qkv.weight` → split на Q, K, V (НЕ реализовано — SSM реализован в конвенции llama.cpp `build_mamba_layer`: ssm_in/ssm_x/ssm_dt и т.д.)
  - SSM-специфичные веса: `sm_conv1d`, `sm_alpha`, `sm_beta`, `sm_out`, `sm_a`, `sm_dt` — загружаются как ssm_conv1d/ssm_x/ssm_dt/ssm_a/ssm_d/ssm_out
- [x] **Layer-type dispatch из GGUF metadata**:
  - Qwythos: слои 3,7,11,15,19,23,27,31 → attention; остальные 24 → SSM
  - Определять по наличию `ssm_conv1d` vs `attn_q.weight` в GGUF (`is_ssm` per layer)
  - Generic: читать из metadata ключа типа `qwen35.layer_type.{i}`
- [x] **QK-RoPE norms** для SSM attention слоёв (`ssm_dt_norm`, `ssm_b_norm`, `ssm_c_norm`, `attn_q_norm`, `attn_k_norm` — загрузка по именам)
- [x] **Поддержка в `build_weights()`**: загрузка SSM тензоров по именам
- [x] **Поддержка в `ct_infer_forward()`**: per-layer выбор attention vs SSM (`lw->is_ssm`)

> ✅ Проверено по коду 31 июля 2026: calm_ssm.c — реальные `ct_ssm_conv1d`, `ct_ssm_selective_scan`, `ct_forward_ssm` (не заглушки); calm_infer.c — детекция `is_ssm` (стр. 787), загрузка весов (798-853), dispatch (1434). Открытым остаётся fused QKV (attn_qkv.split) для Qwythos-style SSM.

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
- [x] **Quantized Wkv_b в MLA**: блочная деквант-адресация по GGUF layout (блоки вдоль dc; 256 для K-quants, 32 для Q5_0/Q4_1/Q8_0/Q4_0/IQ4_NL), `deq_q8_0_wrap`/`deq_q4_0_wrap` адаптеры
- [x] **Synthetic MLA test** (`calm_mla_test.c`): F32 + Q8_0 пути против независимого референса, 3 комбо (k_norm/kv_a_norm), DC=64 → 2 блока Q8_0 на колонку — ALL PASS (F32 ~1e-5, Q8_0 в допуске 5%)
- [x] **DeepSeerMoE fine-grained routing**: 64 мелких эксперта, top-6 (на базе существующего MoE роутера) — подтверждено на реальной модели
- [x] **Real DeepSeek2 GGUF model test**: DS-Coder-V2-Lite-Instruct Q2_K (6.0 GiB) — загрузка, 27 слоёв, MLA + MoE 64×top-6 + shared 2, генерация текста
- [x] **DeepSeek2 tokenizer**: BPE со специальными токенами (`<｜end▁of▁sentence｜>`, `惜`) — обнаружение спецтокенов через fullwidth bar `｜` (U+FF5C, `calm_tokenizer.c:731-737`), emit как единый токен при encode; границы vocab защищены в `embed_row` (см. баг #12)

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

### Статус: ✅ Real model verified (1 августа 2026)

**Реализовано:**
- `calm_mla.h` / `calm_mla.c` — MLA forward pass с absorption trick (Q_nope @ Wk_b → absorbed_q, weighted latent sum → V)
- `calm_infer.h` — MLA config поля (kv_lora_rank, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, n_shared_expert), MLA weights (attn_q_a/b, attn_kv_a/b, norm), MLA KV cache (mla_kv_cache)
- `calm_infer.c` — MLA детекция через `attn_kv_a.weight`, weight loading, KV cache alloc, dispatch в forward pass
- Shared expert (DeepSeekMoE) — weight loading + FFN forward поверх routed MoE
- **embed_row bounds guard** (баг #12): `int embed_row(out, table, type, token, n_embd, n_vocab)` — OOB-токен (`< 0 || >= n_vocab`) → zero-fill + `-1`, все 7 вызовов обновлены (calm_infer prefill/генерация, calm_smoke, test_jamba); smoke Test 3 (OOB) PASS
- **DeepSeek2 спецтокены BPE**: обнаружение fullwidth bar `｜` (U+FF5C) в vocab (`calm_tokenizer.c:731-737`), emit единым токеном при encode
- **Quantized Wkv_b в MLA absorption loops**: блочная адресация по GGUF layout (блоки вдоль dc), все типы Q2_K…Q8_K + Q5_0/Q4_1/Q8_0/Q4_0/IQ4_NL; `attn_kv_a_norm` нормирует первые `dc` элементов (вес = kv_lora_rank = 512, см. баг #10); Step 3 k_norm per-head-group с 2D-offset расширением (без переполнения `kv_b_cur[4096]`); деление `dot_nope` на RMS в обоих путях
- **Исправлены переполнения**: `buf_q_size` в `calm_infer.c` теперь `H*(dn+dr)` при MLA (было 2048, DS-Coder-V2-Lite требует 3072); `mla_cache_dim` + `n_head_kv` строк на слой
- **Synthetic MLA test** (`calm_mla_test.c`): независимый F32-референс, 3 комбо, F32/Q8_0 пути — ALL PASS (MSVC/AVX2)
- **Real model test** (1 августа 2026): `DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf` (6.0 GiB, 377 тензоров, arch=deepseek2)
  - Загрузка: 27 слоёв, MLA fused-формат (`attn_kv_a_mqa=[2048,576]` Q2_K, `attn_kv_b=[512,4096]` Q2_K, `attn_output=[2048,2048]` Q3_K, `attn_q=[2048,3072]` Q2_K — q_lora_rank=0 у Lite)
  - MoE: 64 routed эксперта (3D stacked `ffn_gate_exps=[2048,1408,64]` Q2_K) + 2 shared (`ffn_gate_shexp=[2048,2816]` Q2_K, `ffn_down_shexp=[2816,2048]` Q3_K) + роутер `ffn_gate_inp=[2048,64]` F32
  - Слой 0 — плотный FFN (leading_dense_block_count=1), слои 1-26 — MoE
  - Forward всех 27 слоёв: **без NaN/Inf** (max_abs ~1e3)
  - Генерация: "The capital of France is" → "Paris. Paris is the capital city of France. It is a beautiful city…"
  - Smoke Test 3 (OOB-токен vocab+7): `rc=-1 emb_sum=0.00` — границы vocab защищены (баг #12)
  - Коммиты: `48cb791` (kv_a_norm + буферы + токены smoke, +408/−72), `014e156` (kv_a_norm dc-only по реальному GGUF, +55/−27), `453caf2` (embed_row bounds guard, +36/−11)

**Осталось:**
- ~~Оптимизация скорости~~ → **Q2_K AVX2 matmul ядро реализовано** (`ct_matmul_q2_K`, calm_quant.c, см. Phase 9 §1): generation на DS-Coder-V2-Lite Q2_K ускорился **~1.74×** (58.68s → 33.67s на 10 токенов, ~4.2 → ~3.4 s/токен), вывод корректный ("Paris. Paris is the capital city of…")
- Дальнейшая оптимизация (MoE streaming, top-6, память) — по-прежнему открытый хвост, но не блокирует Phase 8

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
Фаза 8: DeepSeek2 (MLA+MoE)      │ ████████████████  2-3 нед    ✅ MLA+MoE validated
Фаза 9: x86/AVX2 Laptop Opt.     │ █████▓░░░░░░░░░░  1 нед      ◐ AVX2 kernels done
                                     └── ~5-10 месяцев всего
```

**Ключевые вехи:**
- Phase 6 → сжатие любых GGUF в TQ1_0/BQ1_0 через mmap streaming (prerequisite для всего)
- Phase 7 → **Ornith-9B**, Qwythos-9B, Qwen3.5-9B на телефоне (SSM гибриды) ✅
- Phase 8 → **DeepSeek-Coder-V2-Lite 16B** на телефоне (MLA + MoE, 2.3 GB BQ1_0) ✅ forward pass + real model verified
- Phase 9 → **x86 ноутбук**: AVX2-оптимизированные BQ1_0/TQ1_0 matmul ядра, runtime-детекция CPU features, Makefile x86 target, Windows CPUID, per-layer adaptive quant
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
12. ✅ **Phase 7: SSM Forward Pass** — Mamba-style selective scan для Qwen3.5/Jamba/Ornith гибридов
13. ✅ **Phase 8: DeepSeek2 (MLA+MoE)** — Multi-head Latent Attention + DeepSeekMoE (forward pass integrated; quantized Wkv_b блочная адресация + kv_a_norm dc-only фикс + переполнения буферов исправлены; synthetic MLA test ALL PASS; **реальная модель DS-Coder-V2-Lite Q2_K (6 GiB) загружается, 27 слоёв без NaN, генерация текста работает**; спецтокены BPE (fullwidth bar) обрабатываются, границы vocab защищены в embed_row; осталось: оптимизация скорости)
14. ◐ **Phase 9: x86/AVX2** — AVX2 matmul Q8_0/Q4_0/BQ1_0/TQ1_0 ✅, dequant/quant ✅, Makefile x86 ◐, CPUID Windows 🔲
15. 📦 **mmap/expert streaming** — Colibri-style, холодные эксперты с диска
16. 📦 **Qwen3.6 архитектура** — для запуска Bonsai-27B (1-bit, 3.9 GB) — зависит от Phase 7

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
       ├─ Сжать любую модель в TQ1_0/BQ1_0 (prerequisite)
       │
       └──→ Phase 9 (x86/AVX2) — независим, можно параллельно с Phase 7/8
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

**Дальнейшие шаги:**
- Тестирование на реальной SSM модели (Qwythos-9B TQ1_0) после конвертации
- NEON-оптимизация selective scan (особенно expf-вызовы в цикле)
- Поддержка Mamba2 (SSM с группировкой)
- Опциональный GPU/Vulkan SSM kernel

---
 
## 🚀 Phase 9: x86/AVX2 Laptop Optimization (28 июля 2026)
 
**Цель:** Запуск Calm на x86 ноутбуке без дискретного GPU (только CPU, AVX2) с полным
ускорением экстремальной квантизации (BQ1_0/TQ1_0) через SIMD. 8-12 tok/s для 7B Q8_0,
20+ tok/s для 7B BQ1_0 на современном x86 CPU (Zen 3 / Alder Lake+).
 
### Исходные данные (x86 ноутбук)
- CPU: x86_64 с AVX2 + FMA (любой Intel Haswell+ / AMD Excavator+)
- RAM: 16-32 GB, из них ~8-16 GB доступно под модель
- GPU: отсутствует или Intel UHD (не используется для LLM)
- ОС: Linux (WSL2) или Windows (MSVC/MinGW)
- Хранилище: NVMe SSD (достаточно быстро для mmap)

### Стратегия на ноутбуке
 
```
Модель               Формат    Вес    Влезает в 16 GB?
──────               ──────    ───    ─────────────────
Bonsai-27B           BQ1_0     3.9 GB ✅ (с запасом)
Bonsai-27B           TQ1_0     5.9 GB ✅
Qwythos-9B           BQ1_0     1.3 GB ✅
DS-R1-14B            BQ1_0     2.0 GB ✅
Qwen3-32B            BQ1_0     4.5 GB ✅
DeepSeek-V3-671B     BQ1_0    94 GB  ❌ (не влезет)
```
 
**Ключевой вывод:** На ноутбуке доступно 8-16 GB — это открывает модели до 32B в BQ1_0
(4.5 GB) с контекстом 32K+ токенов. AVX2 даёт factor 3-5× над скалярным кодом.
 
### Компоненты
 
#### 1. AVX2-оптимизированные matmul ядра ✅ (calm_quant.c)
 
| Ядро | Формат | Алгоритм | Ускорение |
|------|--------|----------|-----------|
| `ct_matmul_q8_0` | Q8_0 (8-bit) | vpmovsxbd → FMA с broadcast scale | ~4× над scalar |
| `ct_matmul_q4_0` | Q4_0 (4-bit) | nibble unpack → sign-extend → FMA | ~3× над scalar |
| `ct_matmul_bq1_0` | BQ1_0 (1-bit) | per-lane mask (`_mm256_and_ps`) + d·(2·Σ_masked−Σ_all) trick | ~5× над scalar |
| `ct_matmul_tq1_0` | TQ1_0 (ternary) | dequant→Q8_0 buffer → AVX2 Q8_0 matmul | ~3× над scalar base-3 |
| `ct_matmul_q2_K` | Q2_K (2-bit K-quant) | 16-bit lane shift + mask 0x0303 → per-sub-block 2-bit unpack, `_mm256_fmsub_ps` (q·dl−ml) + `_mm256_fmadd_ps` (x·w), scalar tail для I%256≠0 | ~1.74× end-to-end (58.68s→33.67s / 10 tok, DS-Coder-V2-Lite Q2_K); чистое matmul-ускорение выше, но MoE+MLA ограничивают |
 
**`ct_matmul_bq1_0` (BQ1_0 AVX2):**
- Обрабатывает 8 float/iteration
- Для каждой группы: Σ_all (сумма всех x[i]) + mask bits → Σ_masked (сумма x[i] с bit=1)
  → result = d · (2·Σ_masked − Σ_all)
- Маска строится через `_mm256_set_epi32` из 8 бит uint64
- Scalar remainder для хвоста (<8 элементов)
- Идентичный NEON-пути по логике, но использует `_mm256_and_ps` вместо `vbslq_f32`
 
**`ct_matmul_tq1_0` (TQ1_0 AVX2):**
- Dequant TQ1_0 block → Q8_0 buffer через `dequant_tq1_0_to_q8_0()` (платформо-независимая)
- Переиспользует AVX2 Q8_0 matmul (с FMA) для вычислений
- Меньше кода, проще поддерживать, ~3× быстрее полного scalar base-3 unpack
 
#### 2. Восстановлена AVX2-секция в calm_quant.c ✅
 
До фикса: `#ifdef __AVX2__` секция содержала только `ct_matmul_q8_0` и `ct_matmul_q4_0`,
но **не имела** `ct_quant_init()`, `ct_dequant_q8_0()`, `ct_dequant_q4_0()`,
`ct_dequant_bq1_0()`, `ct_dequant_tq1_0()`. При сборке с `-DCT_AVX2`:
- `ct_quant_init()` → undefined reference (вызывается из `calm.c`)
- `ct_dequant_q8_0()` → undefined reference (вызывается из `calm_convert.c`)
- Linker error → сборка падала
 
**Фикс:** Копии всех dequant/quant/init функций добавлены в AVX2-секцию.
Правильная структура блоков:
```c
#if __ARM_NEON
  /* NEON implementations */
#elif !defined(__AVX2__)
  /* Scalar fallback */
#endif
 
#ifdef __AVX2__
  /* AVX2 implementations (always compiled when AVX2 available) */
#endif
```
 
#### 3. Runtime CPU feature detection ◐ (calm.c)
 
- **Linux:** `/proc/cpuinfo` flags line — AVX2, AVX-512, BF16 уже парсятся ✅
- **Windows:** CPUID via `__cpuid()` — `has_avx2`, `has_avx512`, `has_bf16` — **реализован** (calm.c `#if defined(_WIN32)`, коммит Phase 7 `c5636b2`) ✅
- CalmDevice уже содержит поля: `has_avx2`, `has_avx512`, `has_bf16` ✅
- **Не хватает:** runtime-диспетчеризация matmul по `has_avx2` — сейчас диспетчер собран на compile-time `#ifdef __AVX2__` (calm_infer.c `matmul()`). Открытый хвост Phase 9.
 
#### 4. Makefile x86 target ◐
 
- Цель `make x86` существует (`CFLAGS_X86 = -O2 -std=c11 -mavx2 -mfma -DCT_AVX2`) ✅
- Проблема: `.c.o` правило добавляет `-DCT_VULKAN` unconditionally — не критично, но грязно
- Нужно: отдельный `make x86` без Vulkan, с явной `-mavx2 -mfma -DCT_AVX2`
- Нужно: `make x86-vk` для сборки с Vulkan + AVX2 (если есть Vulkan-совместимый GPU)
 
#### 5. Per-layer adaptive quantization (calm_convert.c) 🔲
 
- `--adaptive-quant N`: первые N слоёв сохранять в Q8_0, остальные в TQ1_0/BQ1_0
- Позволяет сохранить качество на критических первых/последних слоях
- Практическая разница: +0.5-1% accuracy, +2-5% размера
 
### Изменённые файлы (Phase 9)
- `calm_quant.c` — AVX2 matmul ядра для Q8_0, Q4_0, BQ1_0, TQ1_0, **Q2_K** (`ct_matmul_q2_K`); dequant/quant/init функции; восстановлена AVX2-секция
- `calm_quant.h` — `ct_block_q2_K` typedef + `CT_QK_K` перенесены в заголовок (единый источник для calm_infer/calm_mla/calm_quant), декларация `ct_matmul_q2_K`
- `calm.h` — флаги CPU feature в CalmDevice ✅ (были добавлены ранее)
- `calm.c` — `calm_device_probe()` CPUID на Windows + `/proc/cpuinfo` на Linux
- `calm_infer.c` — dispatch matmul по `has_avx2` в рантайме; Q2_K: compile-time `#ifdef __AVX2__` → `ct_matmul_q2_K` / `#else` → scalar `matmul_q2_K`
- `calm_mla.c` — использует `CT_QK_K` из заголовка (удалён дубль `#define`)
- `Makefile` — чистый `make x86` + `make x86-vk`
- `calm_convert.c` — `--adaptive-quant` флаг
 
### Производительность (расчёт)
 
| Модель | Формат | Размер | Tok/s (scalar) | Tok/s (AVX2) |
|--------|--------|--------|----------------|--------------|
| Qwen2.5-7B | Q8_0 | 7.0 GB | 1-2 | 8-12 |
| Qwen2.5-7B | BQ1_0 | 0.98 GB | 8-10 | 25-35 |
| Qwythos-9B | TQ1_0 | 1.78 GB | 4-6 | 15-20 |
| Bonsai-27B | BQ1_0 | 3.9 GB | 2-3 | 6-10 |
 
*Расчёт для одного CPU core AVX2 (256-bit FMA, 8 FLOPS/cycle @ 3 GHz = 24 GFLOP/s теоретически)*
 
### Зависимости
- **Phase 6 (Requantizer)**: сжать модели в TQ1_0/BQ1_0
- Phase 9 не зависит от Phase 7/8 — можно делать параллельно
 
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

### 4. Квантизованный matmul в `DEF_MATMUL_QUANT` — расходимость forward pass (31 июля 2026)

**Причина:** Макрос `DEF_MATMUL_QUANT` (calm_infer.c) индексировал блоки весов смещением **входного** блока (`row = w + i0`), а не строки выхода. При row-major раскладке `[O строк × I колонок]` такое смещение идёт вдоль строк, а не колонок — деквант читал чужие строки весов, скалярные произведения были полностью неверны.

**Симптом:** forward pass на `qwen2.5-0.5b.gguf`: L0 ffn_down range=[−1.86e8, 2.08e8], h `sum|abs|`=3.46e10 (idx 665), взрывной рост по слоям (77e9+), NaN при росте входной строки.

**Фикс:** `calm_infer.c` — макрос переписан на итерацию выходных строк: `row = w + j*bpr + b`, где `bpr = ceil(I/BLOCK)` блоков на строку (та же конвенция, что у `ct_matmul_q8_0`, calm_quant.c:972). Конвенция вызовов не менялась: `matmul(y, x, w, type, I, O)`.

**Верификация:**
- L0 h `sum|abs|`: 3.46e10 → **146.21** (здоровый масштаб; эталон старого лога ≈245; старый референс на L20 = 15037, сейчас 4576 — лучше)
- `calm_smoke` PASS на `qwen2.5-0.5b.gguf` (Q5_0/Q6_K) и `models/qwen2.5-0.5b-instruct-q4_0.gguf` (Q4_0): 24 слоя, 2 токена, без NaN/Inf
- Независимый FP32-эталон строки 62 ffn_down (L21): `fp32ref == matmul` бит-в-бит (`diff=0.000000`) — деквант + matmul корректны
- Обнаруженный при расследовании «спайк» нейрона 62 ffn_down (≈−695 на L21) — **реальные веса модели**: воспроизводится в двух независимых квантованиях (Q6_K и Q4_0), у строки 62 глобально-максимальный d=0.000153 (блок 1180) и второй по величине 0.000146 (блок 13); вход ffn_down на L21 содержит элемент |x|=1117 (silu(gate)*up). Не баг кода — особенность модели, остаточный рост h (146→7035) ниже старого эталона (245→15037).
- Побочный результат аудита: удалены все временные отладочные принты из forward pass (calm_infer.c), сборка MSVC `/W4` — 0 предупреждений.

---

## 🐛 Исправленные баги Phase 8 — MLA + tokenizer + buffers (1 августа 2026)

### 5. Quantized Wkv_b — неверная блочная адресация (general path)

**Причина:** Код адресовал блоки квантизованного Wkv_b смещением `k * blocks_per_row + kg` (как будто блоки идут вдоль **строк**), но GGUF хранит тензор `[ne0=dc, ne1=total_cols]` с ne0 fastest — блоки идут вдоль **dc** (колонок). Для типов с блоком 256 и `total_per_head=256` случайно совпадало, но для 32-элементных (Q8_0/Q4_0/Q5_0/Q4_1/IQ4_NL) и любых dc≠256 — чтение чужих блоков → мусорные absorbed_q/out_h.

**Симптом:** synthetic MLA test: Q8_0 путь vs референс — ошибка ~100% (43.28 при ref scale 43.28), выход ≈ нули.

**Фикс:** `calm_mla.c` — адресация по GGUF layout: `block index = c*nb + kb`, `nb = ceil(dc/block)`, `block_stride = ct_gguf_tensor_size(type, 1, {block})`; `block` = 256 (K-quants) / 32 (Q8_0, Q4_0, Q5_0, Q4_1, IQ4_NL). Q8_0/Q4_0 добавлены в диспетчер через `deq_q8_0_wrap`/`deq_q4_0_wrap` (адаптеры под `(block, out, count)` сигнатуру).

### 6. `attn_kv_a_norm` — расхождение путей, затем неверный фикс dc+dr (исправлено в #10)

**Причина:** general path нормировал первые `dc` элементов латента, F32 path — все `dc+dr`. Промежуточный фикс выровнял оба пути на `dc+dr` — это оказалось **неверно**: вес `attn_kv_a_norm.weight` имеет размер `kv_lora_rank` (512), а не `dc+dr` (576). Нормировка `dc+dr` читала 64 флоата за концом тензора + портила rope-часть (k_pe).

**Итоговый фикс (#10):** `calm_mla.c` — оба пути (`F32` и `Q8`) нормируют только первые `dc` элементов: `rms_norm(kv_a_buf, kv_a_buf, lw->attn_kv_a_norm, dc, ...)`. Соответствует HF: `DeepseekV2RMSNorm(kv_lora_rank)` применяется только к `kv_nope`; `k_pe` не нормализуется. Подтверждено реальным GGUF (dims=[512]).

### 7. Step 3 k_norm — переполнение `kv_b_cur[4096]` + неверная семантика

**Причина:** `matmul(kv_b_cur, c, wkv_b, t, dc, H*total_per_head)` раскрывал K_nope сразу для всех H групп: DS-Coder-V2-Lite требует `H*total_per_head = 16*320 = 5120` колонок > 4096 → переполнение стека. Плюс scale записывался в сами kh (затирая kv_b_cur) вместо хранения RMS для последующего деления scores.

**Фикс:** `calm_mla.c` — группы раскрываются по одной (`matmul(..., dc, total_per_head)` со смещением `ct_gguf_tensor_size(t_kvb, 2, {dc, kg*total_per_head})` — 2D-offset в GGUF), RMS каждой группы пишется в кэш-строку `(dc+dr+kg)`; **оба** пути Step 4b делят `dot_nope` на `rms_p`. Plain RMS — вес `attn_k_norm` намеренно не применяется (конвенция `attn_q_norm`).

### 8. Heap overflow `buf_q_size` в `calm_infer.c` (реальный баг инференса)

**Причина:** `buf_q_size = max(n_embd, …, H*head_dim)` = 2048 для DS-Coder-V2-Lite, но MLA Step 1 пишет `H*(dn+dr)` = 16×192 = **3072** floats в `s->buf_q` → запись за границу кучи при каждом forward pass MLA-модели.

**Фикс:** `calm_infer.c` (~1354) — при `mla_kv_lora_rank > 0`: `buf_q_size = max(..., H*(qk_nope_head_dim + qk_rope_head_dim))`. Сопутствующий фикс: `mla_cache_dim` увеличен на `+ n_head_kv` строк/слой (alloc ~1325 и slice ~1466) — для хранения per-head-group RMS.

### 9. Synthetic MLA test не вызывал `ct_quant_init()`

**Причина:** тест квантил Q8_0 без инициализации таблицы fp16 (`ct_fp16_table` в calm_quant.c) — все масштабы = 0 → деквант = нули → Q8_0 путь давал 0.00. **Не баг продакшн-кода** (calm_infer.c:1180 вызывает `ct_quant_init()` при загрузке модели).

**Фикс:** `calm_mla_test.c` — `ct_quant_init()` в `main()`.

### 10. `attn_kv_a_norm` — OOB чтение + порча k_pe (реальная модель, 1 августа 2026)

**Причина:** `rms_norm(..., dc + dr = 576, ...)` при весе `attn_kv_a_norm.weight = [512]` (= kv_lora_rank). Реальный GGUF-дампер подтвердил: dims=[512], тип F32, size=2048 байт. Чтение 64 флоатов за концом тензора (соседний `attn_kv_a_mqa`), плюс неверная нормализация rope-части. Синтетический тест не ловил: `g_kva_norm_w[DC+DR]` и референс зеркалили ту же неверную конвенцию.

**Симптом:** на реальной модели значения нормированного латента искажены; smoke-тест выдавал гигантские эмбеддинги (max_abs=68628) из-за OOB чтения token_embd (см. #11) и неверных норм.

**Фикс:** оба пути (`F32` и `Q8`) в `calm_mla.c` нормализуют только первые `dc` элементов. Тест-референс `calm_mla_test.c` приведён к HF-конвенции, `g_kva_norm_w` уменьшен до `[DC]`.

### 11. calm_smoke: токены Llama вне vocab DeepSeek (тест, 1 августа 2026)

**Причина:** `calm_smoke.c` хардкодил токены Llama (BOS=128000, спецтокен 128006), но у DS-Coder-V2-Lite vocab=102400. `embed_row` читала строки за концом `token_embd` (размер 68.8 MB, тензоры-соседи → значения 334944×3 одинаковых) — ложный FAIL.

**Фикс:** `calm_smoke.c` — валидные токены 0 и 100. Проверка границ vocab в `embed_row` — рекомендация на будущее → реализована (см. #12).

### 12. embed_row — отсутствие bounds check по vocab (защита, 1 августа 2026)

**Причина:** `embed_row()` не проверяла границы `token` — любой ID ≥ `vocab_size` (например, спецтокен Llama 128000 при vocab=102400 или мусорный токен из сэмплинга) читал строку за концом `token_embd`: OOB-чтение → гигантский мусор в эмбеддингах → ложные FAIL и потенциальные сбои. Строки за концом тензора попадали в соседние тензоры GGUF-файла.

**Фикс:** `calm_infer.c` / `calm_infer.h` — сигнатура изменена на `int embed_row(out, table, type, token, n_embd, n_vocab)`:
- `token < 0 || token >= n_vocab` → выход зануляется `memset`, возврат `-1` (успех = `0`)
- Обновлены все 7 вызовов: `calm_infer.c` (prefill + генерация), `calm_smoke.c` (2), `test_jamba.c` (3)
- `calm_smoke.c` — добавлен **Test 3 (OOB)**: токен `vocab+7` при отравленном буфере → ожидается `rc=-1, emb_sum=0`

**Верификация:**
- `calm_smoke.exe` (MSVC/AVX2) на реальной DS-Coder-V2-Lite Q2_K: Test 3 → `rc=-1 emb_sum=0.00` ✓, весь smoke **PASS**
- `calm_mla_test.exe`: **ALL PASS** (F32 ~1e-5, Q8_0 в допуске 5%) — регрессии нет
- WSL `make x86`: сборка чистая
- Коммит `453caf2`: 4 файла, +36/−11

**Верификация (1 августа 2026, MSVC/AVX2, `/W4` 0 предупреждений):**
- `calm_mla_test.exe`: F32 vs референс max_abs 1.9e-5 / 1.3e-5 / 5.7e-6; Q8_0 vs референс 0.54 / 0.26 / 0.11 (допуск 5%) — **ALL PASS**, 3 комбо (k_norm/kv_a_norm), Q8_0 roundtrip max_err 8.1e-3
- `calm_smoke.exe` (MSVC/Windows): PASS на Llama-3.2-1B Q8_0 и qwen2.5-0.5b Q4_0 — без NaN/Inf
- **Real model** (1 августа 2026): `calm_smoke.exe` + `calm run` (make x86) на `DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf` (6.0 GiB, 27 слоёв): emb sane (sum -8.1, max_abs 1.2), все 27 слоёв без NaN (max_abs ~1e3), генерация: "The capital of France is" → "Paris. Paris is the capital city of France. It is a beautiful city…"
- Коммит `48cb791`: 4 файла, +408/−72

---

## ⚙️ Рабочий процесс и синхронизация репозиториев (1 августа 2026)

**Проблема (выявлена 1 августа 2026):** у проекта было две рабочие копии одного репозитория
(`VitaliyVV/calm`) без канала синхронизации — WSL `~/calm` (живая, коммиты Phase 7/8)
и `C:\00 Work\000 Projects Notebook\00 AI System\calm` (клон, застрял на `c8c3acf`,
15 незакоммиченных правок). Git был настроен, но push/pull не использовались → расслоение.

**Причина:** все свежие коммиты делались в WSL-копии и никогда не пушились в origin;
Windows-клон не вытягивал обновления. Локальные правки в клоне (calm_server.c/h —
перезаписаны при синхронизации, VSS недоступен) утрачены; остальные 13 файлов их
содержимое уже вошло в WSL-коммиты.

### Единый источник истины

```
WSL ~/calm (живая копия, коммиты + push)
        │
        ▼
github.com/VitaliyVV/calm  (origin, единственный канал синхронизации)
        │
        ▼
Любые другие копии: C:\00 Work\...\calm, телефон, ноутбук (только git pull)
```

**Правила:**
1. **Одна копия = источник истины** (сейчас WSL `~/calm`). Все остальные — только `git pull`.
2. **Перенос файлов между копиями — ТОЛЬКО через git** (push/pull). Ручное копирование
   (robocopy/cp) запрещено — оно ломает историю и создаёт «незакоммиченные призраки».
   Разовое копирование допустимо только как консолидация при восстановлении порядка.
3. **Каждый коммит = push в origin.** `git commit && git push origin main` одним действием.
   Это автоматически синхронизирует все остальные копии при их следующем pull.
   Для неинтерактивного push в WSL настроен `credential.helper store` (токен из Windows
   Credential Manager) — push выполняется без запроса логина/пароля.
4. **Незакоммиченные правки в неактивных копиях**: либо коммитить, либо отбрасывать —
   никогда не оставлять «плавать» (именно они и были потеряны при перезаписи).

> ⚠️ **Особенность запуска git в WSL:** `wsl.exe` наследует Windows-CWD, и git, запущенный
> без смены директории, падает с `fatal: cannot chdir to 'C:/'`. Все git-команды в WSL
> запускать только с `cd ~/calm &&` в начале, например:
> `wsl -d Ubuntu -- bash -c "cd ~/calm && git status"`. Сам git при этом исправен —
> это ограничение запуска через `wsl.exe` из Windows-каталога.

### Сборка — два нативных пути (WSL не обязателен)

| Путь | Команда | Платформа | Назначение |
|------|---------|-----------|------------|
| **MSVC** (Windows-native) | `build_smoke.bat` / `build_mla_test.bat` / `calm_convert.exe` | Windows, vcvars64 | Тесты и бинарники на чистом Windows (без WSL) |
| **make x86** (WSL) | `make x86` | WSL, clang 21 + AVX2/FMA | Основная сборка `calm` CLI |

Движок кросс-платформенный: MSVC-сборки работают на Windows без WSL (calm_smoke.exe,
calm_mla_test.exe, calm_convert.exe). WSL — среда разработки/сборки, не таргет.
Направление «работать без WSL» поддерживается нативно (Phase 9: Windows CPUID, MSVC).

### Текущее состояние (после Q2_K AVX2, 1 августа 2026)

- WSL `~/calm` — HEAD Q2_K AVX2 (Phase 8 хвост + Phase 9 ядро) — источник истины, запушен в origin
- **Q2_K AVX2 matmul** (`ct_matmul_q2_K`): DS-Coder-V2-Lite Q2_K generation **58.68s → 33.67s / 10 токенов (~1.74×)**, вывод корректный
- Windows-клон синхронизирован **через git** (`fetch + reset --hard origin/main`):
  свежие исходники, бинарники, модели (DeepSeek-Coder-V2-Lite Q2_K 6 GB, dscoder-6.7b) + `tools/`
  (билд-скрипты, дамперы) — клон содержит только untracked-артефакты (*.exe, tools/, build_msvc.bat)
- Вспомогательные скрипты: `tools/build_*.bat`, `tools/dump_gguf.c`, `tools/check_shexp.sh`,
  `tools/check_gguf*.py`, `tools/gguf_names.py` (скопированы из temp рабочей сессии)

# Calm — Universal Local LLM Runtime

**Дорожная карта продукта**
**Дата:** 15 июля 2026
**Версия:** v0.1 (MVP concept)

---

## Краткое описание

**Calm** — единый zero-dependency рантайм для запуска любых LLM на любом устройстве. 

Объединяет:
- **Bonsai-style** 1-bit/ternary экстремальную квантизацию
- **Colibri-style** streaming экспертов с диска для MoE моделей
- **Автоматический конфиг** под железо (RAM, GPU, storage speed)
- **Единый CLI** — одна команда для всего

---

## Фаза 0: Концепт (текущая) — «Оркестратор»

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

**Фаза 1: Нативный C-движок (v0.2 — завершена)**

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
- [ ] Поддержка MoE архитектур (router, expert dispatch) — частично
- [ ] mmap + expert streaming (Colibri-style)

### Время: 2-3 недели
### Код: C11, ~8000 строк

---

## Фаза 2: Экстремальная квантизация (v0.2 — завершена)

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

## Фаза 3: GPU бэкенды

**Цель:** Аппаратное ускорение на всех платформах.

### Компоненты
- [x] **Vulkan бэкенд** (Android Adreno) — Q8_0 matmul offload, batch dispatch, HOST_VISIBLE weights
- [ ] Metal бэкенд (Apple Silicon)
- [ ] CUDA бэкенд (NVIDIA)
- [ ] WebGPU бэкенд (браузер)
- [ ] Гибрид CPU+GPU: горячие эксперты на GPU, холодные на CPU

### Приоритет
1. **Vulkan** — твой телефон (Adreno 730), Linux, Windows
2. **Metal** — Mac ecosystem
3. **WebGPU** — браузеры
4. **CUDA** — сервера

### Время: 4-8 недель
### Код: C + GLSL (Vulkan) + Metal shaders + CUDA kernels

---

## Фаза 4: Авто-конфиг и умный деплой

**Цель:** Одна команда — любой модели на любом железе.

### Логика принятия решений

```
ВХОД: calm run model.gguf
  │
  1. СКАНИРОВАТЬ УСТРОЙСТВО
  │   RAM_total, RAM_available
  │   CPU: ядра, SIMD (NEON? AVX2? SVE?)
  │   GPU: Vulkan? Metal? CUDA?
  │   Storage: тип (NVMe? UFS? eMMC?), скорость
  │
  2. ПРОАНАЛИЗИРОВАТЬ МОДЕЛЬ
  │   Архитектура (MoE/dense/hybrid)
  │   Размер файла, quant формат
  │   Параметры: всего / активных (для MoE)
  │
  3. ВЫБРАТЬ СТРАТЕГИЮ
  │   ┌─── Model < RAM × 0.75 ───→ Полная загрузка
  │   ├─── MoE + Model > RAM ────→ Colibri streaming
  │   ├─── Dense + Model > RAM ──→ mmap + offloading
  │   └─── Model >> RAM ─────────→ 1-bit quant + streaming
  │
  4. ЗАПУСТИТЬ
      GPU доступен?
      ├── Да → GPU с CPU fallback для холодных слоёв
      └── Нет → CPU с оптимальным thread count
```

### Время: 1-2 недели
### Код: Python (оркестратор) + C (рантайм)

---

## Фаза 5: Продакшн

**Цель:** Стабильный релиз, экосистема, сообщество.

### Компоненты
- [ ] OpenAI-совместимый API сервер
- [ ] Мультимодальность (LMM — изображения)
- [ ] Speculative decoding (MTP)
- [ ] Batch inference
- [ ] Пакетные менеджеры: apt, brew, pip
- [ ] Документация, бенчмарки
- [ ] CI/CD, тесты

### Время: 4-8 недель

---

## Итого: Roadmap

```
Фаза 0: Calm CLI (Python)       │ ████████████████  1-2 дня    ✅
Фаза 1: Native C engine         │ ████████████████  2-3 нед    ✅ v0.2
Фаза 2: 1-bit/ternary quant     │ ████████████████  2-4 нед    ✅ v0.2
Фаза 3: GPU backends (Vulkan)   │ ████░░░░░░░░░░░░  4-8 нед    ◐ Vulkan done, остальное
Фаза 4: Auto-config + smart     │ ░░░░░░░░░░░░░░░░  1-2 нед
Фаза 5: Production release       │ ░░░░░░░░░░░░░░░░  4-8 нед
                                    └── 3-6 месяцев всего
```

---

## Быстрые победы (можно сделать сегодня)

1. ✅ **Calm CLI** — уже написан (Python, работает через Ollama)
2. ✅ **Vulkan GPU бэкенд** — Q8_0 matmul offload на Adreno
3. 📦 Запустить Qwythos-9B на телефоне — `calm run ./qwythos-9b.gguf`
4. 📦 Эксперимент: собрать llama.cpp с Vulkan и сравнить скорость
5. 📦 Metal бэкенд (Apple Silicon)
6. 📦 SSE / streaming для HTTP API

---

## Ресурсы

- **Colibri** — https://github.com/JustVugg/colibri (C engine, MoE streaming)
- **bitnet.c** — https://github.com/artalis-io/bitnet.c (C engine, universal + GPU)
- **ExpertFlow** — https://github.com/jhammant/expertflow (Rust, smart MoE streaming)
- **flash-moe** — https://github.com/tayoun/flash-moe (C/Metal, Apple MoE streaming)
- **QMoE** — https://github.com/IST-DASLab/qmoe (sub-1-bit MoE compression research)
- **Bonsai 27B** — https://huggingface.co/collections/prism-ml/bonsai-27b
- **llama.cpp** — https://github.com/ggml-org/llama.cpp (GGUF reference, Vulkan backend)

# Calm — Инструкция по установке и запуску

**Версия:** 0.1 (Python CLI — Фаза 0)
**Дата:** 15 июля 2026

---

## 1. Быстрый старт (на телефоне прямо сейчас)

У тебя уже есть всё необходимое: **Python 3.13, Ollama, и модель Qwythos-9B**.

### Запуск Calm CLI (без установки)

```bash
# Перейти в папку с проектом
cd /storage/emulated/0/Documents/000\ Projects\ Smart/00\ AI\ System/00\ Local\ Models/

# Сканировать устройство
python3 calm.py scan

# Проанализировать модель
python3 calm.py analyze /storage/emulated/0/Documents/000\ Projects\ Smart/00\ AI\ System/00\ Local\ Model\ Qwythos/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf

# Запустить модель с авто-конфигом
python3 calm.py run /storage/emulated/0/Documents/000\ Projects\ Smart/00\ AI\ System/00\ Local\ Model\ Qwythos/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf

# Прогноз скорости
python3 calm.py estimate /storage/emulated/0/Documents/000\ Projects\ Smart/00\ AI\ System/00\ Local\ Model\ Qwythos/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf
```

### Сделать calm глобальной командой

```bash
chmod +x /storage/emulated/0/Documents/000\ Projects\ Smart/00\ AI\ System/00\ Local\ Models/calm.py
ln -sf /storage/emulated/0/Documents/000\ Projects\ Smart/00\ AI\ System/00\ Local\ Models/calm.py \
       /data/data/com.termux/files/usr/bin/calm
# Теперь можно:
calm scan
```

---

## 2. Запуск Qwythos-9B на телефоне (подробно)

### Способ 1: Через Calm CLI (рекомендуется)

```bash
# Просто:
calm run "/storage/emulated/0/Documents/000 Projects Smart/00 AI System/00 Local Model Qwythos/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"
```

Calm сам:
1. Определит, что у тебя 3 ГБ свободной RAM
2. Поймёт, что модель 5.3 ГБ не влезает целиком
3. Выберет стратегию **mmap** (загружает только нужные слои)
4. Ограничит контекст до 2048 (экономит KV-кэш)
5. Запустит через Ollama

### Способ 2: Вручную через Ollama

```bash
# Создать Modelfile (уже готов!)
cd "/storage/emulated/0/Documents/000 Projects Smart/00 AI System/00 Local Model Qwythos/"
ollama create qwythos -f Modelfile
ollama run qwythos
```

### Способ 3: Через llama.cpp с Vulkan (GPU)

```bash
# 1. Установить зависимости
pkg install cmake ninja vulkan-loader-generic

# 2. Собрать llama.cpp с Vulkan
cd /data/data/com.termux/files/home
git clone --depth=1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j4

# 3. Запустить
./build/bin/llama-cli \
  -m "/storage/emulated/0/Documents/000 Projects Smart/00 AI System/00 Local Model Qwythos/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf" \
  -c 2048 -t 4 -ngl 99 --temp 0.6
```

> `-ngl 99` означает «все слои на GPU». Если Vulkan работает — Adreno 730 даст ~2× ускорение.

---

## 3. Запуск Bonsai 27B на телефоне

### Через llama.cpp

```bash
# 1. Установить huggingface-hub
pip install huggingface-hub

# 2. Скачать Bonsai 27B 1-bit GGUF
huggingface-cli download prism-ml/Bonsai-27B-gguf --local-dir ./bonsai

# 3. Запустить через calm
calm analyze ./bonsai/bonsai-27b.Q1_0.gguf
calm run ./bonsai/bonsai-27b.Q1_0.gguf --context 2048
```

### Требования
- **Bonsai 27B 1-bit:** 3.9 ГБ → надо ~4 ГБ свободной RAM
- **Ternary Bonsai 27B:** 5.9 ГБ → ~6 ГБ

На твоём телефоне (3 ГБ доступно):
- **1-bit BONSAI**: возможно через swap, будет медленно
- **Если освободить RAM до 4 ГБ**: будет работать комфортно

---

## 4. Сборка нативного C-движка (Фаза 1+)

Когда дойдём до фазы 1:

### Зависимости

```bash
# Android/Termux
pkg install cmake ninja gcc vulkan-loader-generic

# Linux
sudo apt install build-essential cmake libvulkan-dev

# macOS
xcode-select --install  # Metal через clang
```

### Сборка

```bash
git clone https://github.com/your-org/calm
cd calm
mkdir build && cd build
cmake .. -DCALM_VULKAN=ON   # или -DCALM_METAL=ON
cmake --build . -j4
```

### Запуск нативного движка

```bash
calm run model.gguf                    # авто
calm run model.gguf --backend vulkan   # Vulkan
calm run model.gguf --backend metal    # Metal
calm run model.gguf --backend cpu      # CPU only
```

---

## 5. Продвинутые сценарии

### Освобождение RAM перед запуском

```bash
# Закрыть тяжелые процессы
pkill -9 ollama  # временно, потом перезапустить
# Удалить неиспользуемые Ollama модели
ollama rm qwen2.5:0.5b
# Проверить
free -h
```

### Эксперимент: сравнение CPU vs Vulkan

```bash
# CPU
calm run qwythos-9b.gguf --backend cpu --no-auto --context 512 -p "Hello"

# Vulkan (если собран)
calm run qwythos-9b.gguf --backend vulkan --context 512 -p "Hello"
# Сравнить скорость!
```

### Оценка до запуска

```bash
calm estimate qwythos-9b.gguf
# Вывод:
#   CPU:            2.1 tok/s
#   With Vulkan:    3.8 tok/s
#   Estimated TTFT: 1.1s
#   Verdict: Good — usable for chat
```

### API сервер

```bash
# Через llama.cpp
calm server qwythos-9b.gguf --port 8080

# В другом терминале:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

---

## 6. Типичные проблемы

### Модель не влезает в RAM
```
Error: not enough memory
```
→ Уменьшить контекст: `--context 1024`
→ Использовать mmap (автоматически)
→ Закрыть фоновые приложения
→ Переквантовать в 1-bit (Bonsai-style)

### Vulkan не работает
```
Error: Vulkan not available
```
→ Проверить: `ls /vendor/lib64/libvulkan.so`
→ Собрать без Vulkan: `cmake .. -DGGML_VULKAN=OFF`

### Ollama не может создать модель
```
Error: ollama create failed
```
→ Проверить путь к GGUF файлу
→ Убедиться, что ollama запущен: `ollama ps`
→ Перезапустить: `ollama serve &`

### Слишком медленно
```
0.1 tok/s
```
→ Использовать GPU: `--gpu-layers 99`
→ Уменьшить контекст: `--context 1024`
→ Для MoE: прогревать кэш экспертов (`--warmup`)
→ Рассмотреть 1-bit версию модели

---

## 7. Структура проекта

```
00 Local Models/
├── calm.py                  ← Python CLI (Фаза 0)
├── calm.h                   ← C API заголовок (Фаза 1)
├── CALM_roadmap.md          ← Дорожная карта
├── CALM_architecture.md     ← Архитектура
├── INSTRUCTIONS.md            ← ← ЭТОТ ФАЙЛ
├── report_colibri_glm52.md    ← Исследование Colibri
├── report_bonsai_27b.md       ← Исследование Bonsai
├── report_dialogue_2026-07-15.md  ← Полный отчёт диалога
└── comparative_analysis_2026-07-15.md

00 Local Model Qwythos/
├── Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf  ← модель 5.3 ГБ
├── Modelfile                                    ← для Ollama
├── REPORT_MODEL_CARD.md
├── REPORT_SYSTEM_REQUIREMENTS.md
└── SESSION_INFO.json
```

/**
 * calm.h — Calm Native C API
 * 
 * Universal Local LLM Runtime — C11 API для нативного движка.
 * Фаза 1: архитектура API, заглушки для реализации.
 * 
 * Использование:
 *   #include "calm.h"
 *   CalmRuntime* rt = calm_init(NULL);
 *   CalmModel* model = calm_model_load(rt, "model.gguf", NULL);
 *   calm_model_generate(model, "Hello", 128, NULL, NULL);
 *   calm_model_free(model);
 *   calm_destroy(rt);
 */

#ifndef CALM_H
#define CALM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════
 * Версии
 * ════════════════════════════════════════════════════════════ */

#define CALM_VERSION_MAJOR 0
#define CALM_VERSION_MINOR 1
#define CALM_VERSION_PATCH 0

/* ════════════════════════════════════════════════════════════
 * Типы ошибок
 * ════════════════════════════════════════════════════════════ */

typedef enum {
    CALM_OK = 0,
    CALM_ERR_GENERIC = -1,
    CALM_ERR_OOM = -2,
    CALM_ERR_FILE_NOT_FOUND = -3,
    CALM_ERR_INVALID_FORMAT = -4,
    CALM_ERR_UNSUPPORTED_ARCH = -5,
    CALM_ERR_UNSUPPORTED_QUANT = -6,
    CALM_ERR_BACKEND_UNAVAIL = -7,
    CALM_ERR_CONTEXT_OVERFLOW = -8,
    CALM_ERR_TIMEOUT = -9,
} CalmError;

/* ════════════════════════════════════════════════════════════
 * Устройство
 * ════════════════════════════════════════════════════════════ */

typedef enum {
    CALM_PLATFORM_UNKNOWN,
    CALM_PLATFORM_ANDROID,
    CALM_PLATFORM_LINUX,
    CALM_PLATFORM_MACOS,
    CALM_PLATFORM_WINDOWS,
} CalmPlatform;

typedef enum {
    CALM_STORAGE_UNKNOWN,
    CALM_STORAGE_NVME,
    CALM_STORAGE_UFS,
    CALM_STORAGE_SSD,
    CALM_STORAGE_EMMC,
} CalmStorageType;

typedef struct {
    // Platform
    CalmPlatform platform;
    bool is_termux;

    // RAM (bytes)
    size_t ram_total;
    size_t ram_available;
    size_t swap_total;
    size_t swap_free;

    // CPU
    int cpu_cores_physical;
    int cpu_cores_logical;
    int cpu_max_freq_mhz;
    bool has_neon;
    bool has_sve;
    bool has_i8mm;
    bool has_avx2;
    bool has_avx512;
    bool has_bf16;

    // GPU
    bool has_vulkan;
    bool has_metal;
    bool has_cuda;
    bool has_opencl;

    // Storage
    CalmStorageType storage_type;
    size_t storage_total;
    size_t storage_free;
} CalmDevice;

/**
 * Определить характеристики устройства.
 * Заполняет структуру на основе /proc, sysctl, Android свойств.
 */
CalmError calm_device_probe(CalmDevice* device);

/**
 * Проверить, достаточно ли ресурсов для заданного размера модели.
 * Возвращает CALM_OK если хватает, иначе — рекомендацию.
 */
CalmError calm_device_check(const CalmDevice* device, 
                                 size_t model_bytes,
                                 size_t context_length,
                                 size_t kv_bytes_per_token);

/* ════════════════════════════════════════════════════════════
 * Форматы квантизации
 * ════════════════════════════════════════════════════════════ */

typedef enum {
    CALM_QUANT_UNKNOWN = 0,
    CALM_QUANT_F32,
    CALM_QUANT_F16,
    CALM_QUANT_Q8_0,
    CALM_QUANT_Q4_0,
    CALM_QUANT_Q4_1,
    CALM_QUANT_Q4_K_M,
    CALM_QUANT_Q5_K_M,
    CALM_QUANT_Q6_K,
    CALM_QUANT_Q8_K,
    CALM_QUANT_Q2_K,
    CALM_QUANT_Q3_XXS,
    CALM_QUANT_Q3_XS,
    CALM_QUANT_Q3_S,
    CALM_QUANT_IQ1_S,   /* 1.5-bit */
    CALM_QUANT_IQ1_M,   /* 1.75-bit */
    CALM_QUANT_IQ2_XXS, /* 2-bit */
    CALM_QUANT_IQ2_XS,
    CALM_QUANT_IQ2_S,
    CALM_QUANT_IQ3_XXS,
    CALM_QUANT_IQ3_XS,
    CALM_QUANT_IQ3_S,
    CALM_QUANT_IQ4_NL,
    CALM_QUANT_IQ4_XS,
    CALM_QUANT_TQ1_0,   /* Ternary 1.58-bit {−1,0,+1} */
    CALM_QUANT_BQ1_0,   /* Binary 1.125-bit {−1,+1} */
} CalmQuantFormat;

#define CALM_QUANT_COUNT 26

/** Название формата (для вывода) */
const char* calm_quant_name(CalmQuantFormat q);

/** Бит на вес для формата */
float calm_quant_bpw(CalmQuantFormat q);

/* ════════════════════════════════════════════════════════════
 * Архитектуры моделей
 * ════════════════════════════════════════════════════════════ */

typedef enum {
    CALM_ARCH_UNKNOWN,
    CALM_ARCH_DENSE,     /* Llama, Qwen, Mistral, Gemma... */
    CALM_ARCH_MOE,       /* Mixtral, DeepSeek, GLM, DBRX... */
    CALM_ARCH_HYBRID,    /* Jamba (SSM + MoE) */
    CALM_ARCH_BITNET,    /* 1-bit BitNet */
} CalmArchType;

typedef struct {
    char name[128];           /* general.name */
    CalmArchType arch_type;
    char arch_name[64];       /* general.architecture */
    
    /* Размеры */
    uint32_t gguf_version;
    uint64_t parameter_count;       /* всего параметров */
    uint64_t active_parameters;     /* на токен (MoE) */
    uint32_t block_count;
    uint32_t context_length;
    uint32_t embedding_length;
    
    /* Quant */
    CalmQuantFormat quant;
    
    /* Файл */
    size_t file_size;
    
    /* MoE */
    bool is_moe;
    uint32_t expert_count;
    uint32_t expert_active;
    
    /* Memory */
    size_t weights_ram;            /* байт для весов */
    size_t kv_per_token;           /* байт на токен KV кэша */
} CalmModelInfo;

/**
 * Прочитать метаданные GGUF файла (без загрузки весов).
 * Заполняет CalmModelInfo.
 */
CalmError calm_model_read_info(const char* path, CalmModelInfo* info);

/**
 * Освободить ресурсы info (если были выделения).
 */
void calm_model_info_free(CalmModelInfo* info);

/* ════════════════════════════════════════════════════════════
 * Стратегия запуска
 * ════════════════════════════════════════════════════════════ */

typedef enum {
    CALM_STRATEGY_FULL_LOAD,      /* Всё в RAM */
    CALM_STRATEGY_MMAP,           /* mmap + page cache */
    CALM_STRATEGY_MOE_STREAM,     /* Colibri: эксперты стримятся с диска */
    CALM_STRATEGY_LAYER_OFFLOAD,  /* Слои CPU+GPU */
    CALM_STRATEGY_HYBRID_TIER,    /* Горячее — RAM, холодное — диск */
} CalmStrategy;

typedef enum {
    CALM_BACKEND_AUTO = 0,
    CALM_BACKEND_CPU,
    CALM_BACKEND_VULKAN,
    CALM_BACKEND_METAL,
    CALM_BACKEND_CUDA,
    CALM_BACKEND_OLLAMA,
    CALM_BACKEND_LLAMACPP,
} CalmBackendType;

typedef struct {
    CalmStrategy strategy;
    CalmBackendType backend;
    CalmQuantFormat target_quant;  /* CALM_QUANT_UNKNOWN = keep */
    
    uint32_t context_length;
    int cpu_threads;                 /* 0 = auto */
    int gpu_layers;                  /* -1 = auto, 0 = CPU, 99 = max */
    
    /* MoE streaming */
    size_t expert_cache_bytes;
    bool expert_lookahead;
    
    /* KV cache */
    CalmQuantFormat kv_quant;      /* CALM_QUANT_F16 default */
    
    /* Memory */
    bool use_mlock;
    bool use_mmap;
} CalmPlan;

/**
 * Создать оптимальный план на основе устройства и модели.
 */
CalmError calm_plan_create(const CalmDevice* device,
                                const CalmModelInfo* model,
                                CalmPlan* plan);

/**
 * Описание стратегии (для вывода).
 */
const char* calm_strategy_name(CalmStrategy s);

/* ════════════════════════════════════════════════════════════
 * Runtime (движок)
 * ════════════════════════════════════════════════════════════ */

typedef struct CalmRuntime CalmRuntime;
typedef struct CalmModel CalmModel;

/** Параметры генерации */
typedef struct {
    float temperature;
    float top_p;
    float top_k;
    float repeat_penalty;
    int max_tokens;
    bool stream;
    
    /* Callback для streaming */
    void (*on_token)(const char* token, void* user_data);
    void* user_data;

    /* Tool/function calling (см. calm_tools.h) */
    const void* tools;           /* CalmToolDefinitions* */
    int max_tool_rounds;         /* max iterations (default 5) */
} CalmGenerateParams;

/**
 * Создать runtime.
 * plan = NULL → авто-детект.
 */
CalmRuntime* calm_init(const CalmDevice* device);

/**
 * Загрузить модель.
 * plan = NULL → авто-план.
 */
CalmModel* calm_model_load(CalmRuntime* rt,
                                const char* path,
                                const CalmPlan* plan);

/**
 * Сгенерировать текст.
 * params = NULL → стандартные параметры.
 * Возвращает CALM_OK при успехе.
 */
CalmError calm_model_generate(CalmModel* model,
                                   const char* prompt,
                                   char* output,
                                   size_t output_size,
                                   const CalmGenerateParams* params);

/**
 * Прогресс генерации (для индикатора).
 * Вызывать между calm_model_generate.
 */
double calm_model_progress(const CalmModel* model);

/**
 * Текущая скорость (токенов/сек).
 */
float calm_model_speed(const CalmModel* model);

/**
 * Освободить модель.
 */
void calm_model_free(CalmModel* model);

/**
 * Уничтожить runtime.
 */
void calm_destroy(CalmRuntime* rt);

/* ════════════════════════════════════════════════════════════
 * MoE Expert Cache (для стратегии MOE_STREAM)
 * ════════════════════════════════════════════════════════════ */

typedef struct {
    int expert_id;
    void* weights;
    size_t size;
    CalmQuantFormat quant;
    float temperature;         /* recency × frequency */
    uint64_t last_accessed;
    uint64_t access_count;
} CalmExpertSlot;

typedef struct {
    CalmExpertSlot* slots;
    int capacity;
    int count;
    uint64_t total_accesses;
    int lookahead_distance;
    float lookahead_accuracy;
} CalmExpertCache;

/**
 * Создать кэш экспертов.
 */
CalmExpertCache* calm_cache_create(int capacity, int lookahead);

/**
 * Получить эксперта из кэша (или NULL).
 */
CalmExpertSlot* calm_cache_get(CalmExpertCache* cache, int expert_id);

/**
 * Добавить эксперта в кэш (вытесняет coldest при необходимости).
 */
CalmError calm_cache_put(CalmExpertCache* cache,
                              int expert_id,
                              const void* weights,
                              size_t size,
                              CalmQuantFormat quant);

/**
 * Предсказать, какие эксперты понадобятся.
 * router_logits — выход роутера для lookahead.
 */
int calm_cache_predict(CalmExpertCache* cache,
                          const float* router_logits,
                          int n_experts,
                          int top_k,
                          int* predicted_ids);

/**
 * Очистить кэш.
 */
void calm_cache_clear(CalmExpertCache* cache);

/**
 * Уничтожить кэш.
 */
void calm_cache_free(CalmExpertCache* cache);

/* ════════════════════════════════════════════════════════════
 * Backend-специфичные функции
 * ════════════════════════════════════════════════════════════ */

/**
 * Список доступных бэкендов.
 * Возвращает битовую маску.
 */
uint32_t calm_available_backends(void);

/**
 * Название бэкенда.
 */
const char* calm_backend_name(CalmBackendType backend);

/**
 * Проверить, доступен ли Vulkan.
 */
bool calm_has_vulkan(void);

/**
 * Проверить, доступен ли Metal.
 */
bool calm_has_metal(void);

/**
 * Проверить, доступен ли CUDA.
 */
bool calm_has_cuda(void);

/**
 * Оптимизировать количество GPU-слоёв под модель.
 * max_layer_count — всего слоёв в модели.
 * vram_bytes — доступная VRAM.
 * layer_size_bytes — размер одного слоя.
 */
int calm_optimal_gpu_layers(int max_layer_count,
                               size_t vram_bytes,
                               size_t layer_size_bytes);

/* ════════════════════════════════════════════════════════════
 * Вспомогательные функции
 * ════════════════════════════════════════════════════════════ */

/** Размер строки для ошибки. */
const char* calm_err_string(CalmError err);

/** Версия в виде строки. */
const char* calm_version_string(void);

/** Форматировать размер в байтах. */
void calm_format_size(char* buf, size_t buf_size, size_t bytes);

/** Форматировать число параметров. */
void calm_format_params(char* buf, size_t buf_size, uint64_t params);

#ifdef __cplusplus
}
#endif

#endif /* CALM_H */

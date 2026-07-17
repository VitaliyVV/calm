/**
 * calm.c — Calm Native C Engine
 *
 * Universal Local LLM Runtime — Фаза 0/1.
 * Единый zero-dependency C-движок для запуска LLM на любом устройстве.
 *
 * Сборка:
 *   clang -O3 -std=c11 -o calm calm.c -lm
 *   clang -O3 -std=c11 -o calm calm.c -lm -DCALM_VULKAN
 *
 * Использование:
 *   ./calm scan
 *   ./calm analyze model.gguf
 *   ./calm run model.gguf
 *   ./calm estimate model.gguf
 */

#define _GNU_SOURCE
#include "calm.h"
#include "calm_infer.h"
#include "calm_tokenizer.h"
#include "calm_tools.h"
#include "calm_server.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

/* ═══════════════════════════════════════════════════════════════
 * Внутренние константы
 * ═══════════════════════════════════════════════════════════════ */

#define CALM_MAX_PATH 4096
#define CALM_MAX_NAME 256
#define CALM_MAX_STRING 65536
#define CALM_MAX_GPU 8
#define CALM_MAX_METADATA 4096
#define CALM_MAX_CMD 8192
#define CALM_MAX_OUTPUT 65536

/* ═══════════════════════════════════════════════════════════════
 * Вспомогательные функции
 * ═══════════════════════════════════════════════════════════════ */

const char* calm_err_string(CalmError err) {
    switch (err) {
        case CALM_OK: return "OK";
        case CALM_ERR_GENERIC: return "Generic error";
        case CALM_ERR_OOM: return "Out of memory";
        case CALM_ERR_FILE_NOT_FOUND: return "File not found";
        case CALM_ERR_INVALID_FORMAT: return "Invalid format";
        case CALM_ERR_UNSUPPORTED_ARCH: return "Unsupported architecture";
        case CALM_ERR_UNSUPPORTED_QUANT: return "Unsupported quantization";
        case CALM_ERR_BACKEND_UNAVAIL: return "Backend unavailable";
        case CALM_ERR_CONTEXT_OVERFLOW: return "Context overflow";
        case CALM_ERR_TIMEOUT: return "Timeout";
        default: return "Unknown error";
    }
}

const char* calm_version_string(void) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d.%d",
             CALM_VERSION_MAJOR, CALM_VERSION_MINOR, CALM_VERSION_PATCH);
    return buf;
}

void calm_format_size(char* buf, size_t buf_size, size_t bytes) {
    if (bytes >= (size_t)100 << 30) {
        snprintf(buf, buf_size, "%.1f TiB", bytes / (double)((size_t)1 << 40));
    } else if (bytes >= (size_t)100 << 20) {
        snprintf(buf, buf_size, "%.1f GiB", bytes / (double)((size_t)1 << 30));
    } else if (bytes >= (size_t)100 << 10) {
        snprintf(buf, buf_size, "%.1f MiB", bytes / (double)((size_t)1 << 20));
    } else if (bytes >= 1000) {
        snprintf(buf, buf_size, "%.1f KiB", bytes / 1024.0);
    } else {
        snprintf(buf, buf_size, "%zu B", bytes);
    }
}

void calm_format_params(char* buf, size_t buf_size, uint64_t params) {
    if (params >= 1000000000000ULL) {
        snprintf(buf, buf_size, "%.1fT", params / 1e12);
    } else if (params >= 1000000000ULL) {
        snprintf(buf, buf_size, "%.1fB", params / 1e9);
    } else if (params >= 1000000ULL) {
        snprintf(buf, buf_size, "%.1fM", params / 1e6);
    } else {
        snprintf(buf, buf_size, "%" PRIu64, params);
    }
}

static size_t read_file(const char* path, char* buf, size_t buf_size) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(buf, 1, buf_size - 1, f);
    fclose(f);
    buf[n] = '\0';
    return n;
}

static int read_int_from_file(const char* path) {
    char buf[64];
    if (read_file(path, buf, sizeof(buf)) == 0) return -1;
    return atoi(buf);
}

static bool str_ends_with(const char* str, const char* suffix) {
    size_t sl = strlen(str), su = strlen(suffix);
    return sl >= su && strcmp(str + sl - su, suffix) == 0;
}

static bool str_starts_with(const char* str, const char* prefix) {
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

/* ═══════════════════════════════════════════════════════════════
 * 1. Device Probing
 * ═══════════════════════════════════════════════════════════════ */

CalmError calm_device_probe(CalmDevice* device) {
    if (!device) return CALM_ERR_GENERIC;
    memset(device, 0, sizeof(CalmDevice));

    // Платформа
#if defined(__ANDROID__) || defined(ANDROID)
    device->platform = CALM_PLATFORM_ANDROID;
#elif defined(__linux__)
    device->platform = CALM_PLATFORM_LINUX;
    // Проверка Termux
    const char* prefix = getenv("PREFIX");
    if (prefix && strstr(prefix, "com.termux")) {
        device->is_termux = true;
    }
#elif defined(__APPLE__) && defined(__MACH__)
    device->platform = CALM_PLATFORM_MACOS;
#elif defined(_WIN32)
    device->platform = CALM_PLATFORM_WINDOWS;
#endif

    // RAM — /proc/meminfo
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemTotal: %zu kB", &device->ram_total) == 1) {
                device->ram_total *= 1024;
            } else if (sscanf(line, "MemAvailable: %zu kB", &device->ram_available) == 1) {
                device->ram_available *= 1024;
            } else if (sscanf(line, "SwapTotal: %zu kB", &device->swap_total) == 1) {
                device->swap_total *= 1024;
            } else if (sscanf(line, "SwapFree: %zu kB", &device->swap_free) == 1) {
                device->swap_free *= 1024;
            }
        }
        fclose(f);
    }

#if defined(__APPLE__)
    // macOS fallback
    if (device->ram_total == 0) {
        size_t mem_size = 0;
        size_t len = sizeof(mem_size);
        if (sysctlbyname("hw.memsize", &mem_size, &len, NULL, 0) == 0) {
            device->ram_total = mem_size;
        }
    }
#endif

    // CPU — /proc/cpuinfo
    device->cpu_cores_logical = (int)sysconf(_SC_NPROCESSORS_CONF);
    if (device->cpu_cores_logical <= 0) device->cpu_cores_logical = 1;
    device->cpu_cores_physical = device->cpu_cores_logical;

    f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        int impl = -1, part = -1;
        bool has_neon = false;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "CPU part", 8) == 0) {
                unsigned int p = 0;
                if (sscanf(line, "%*[^:]: %x", &p) == 1) {
                    if (part == -1) part = (int)p;
                    else if (part != (int)p) { /* hybrid */ }
                }
            }
            if (strncmp(line, "Features", 8) == 0 || strncmp(line, "flags", 5) == 0) {
                if (strstr(line, "neon") || strstr(line, "asimd")) device->has_neon = true;
                if (strstr(line, "sve")) device->has_sve = true;
                if (strstr(line, "i8mm")) device->has_i8mm = true;
                if (strstr(line, "avx2")) device->has_avx2 = true;
                if (strstr(line, "avx512")) device->has_avx512 = true;
                if (strstr(line, "bf16")) device->has_bf16 = true;
                has_neon = true;
            }
        }
        fclose(f);

        // На Android все ядра big.LITTLE — физические = логические / 2 (грубо)
        if (device->platform == CALM_PLATFORM_ANDROID) {
            int cluster_count = 0;
            DIR* dir = opendir("/sys/devices/system/cpu/cpufreq/");
            if (dir) {
                struct dirent* entry;
                while ((entry = readdir(dir))) {
                    if (entry->d_type == DT_DIR && strncmp(entry->d_name, "policy", 6) == 0)
                        cluster_count++;
                }
                closedir(dir);
            }
            // Каждый policy = кластер. На 8+ gen1 их 3 (X2, A710, A510)
            if (cluster_count > 1)
                device->cpu_cores_physical = cluster_count;
        }
    }

    // CPU freq — ищем макс
    for (int i = 0; i < device->cpu_cores_logical && i < 16; i++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        int freq = read_int_from_file(path);
        if (freq > device->cpu_max_freq_mhz) {
            // Файл даёт kHz, переводим в MHz
            if (freq > 100000) freq /= 1000;
            device->cpu_max_freq_mhz = freq;
        }
    }

    // GPU — проверка Vulkan/OpenCL/Metal
#if defined(__ANDROID__)
    device->has_vulkan = (access("/system/lib64/libvulkan.so", F_OK) == 0) ||
                         (access("/vendor/lib64/libvulkan.so", F_OK) == 0);
    device->has_opencl = (access("/system/lib64/libOpenCL.so", F_OK) == 0) ||
                         (access("/vendor/lib64/libOpenCL.so", F_OK) == 0);
#else
    device->has_vulkan = (system("vulkaninfo --version >/dev/null 2>&1") == 0);
#endif

#if defined(__APPLE__)
    device->has_metal = true; // Все Mac с Apple Silicon или Intel имеют Metal
#endif

    // CUDA — проверка
    device->has_cuda = (system("nvcc --version >/dev/null 2>&1") == 0) ||
                       (system("nvidia-smi >/dev/null 2>&1") == 0);

    // Storage
    struct statvfs vfs;
    const char* storage_path = "/storage/emulated/0";
    if (statvfs(storage_path, &vfs) != 0)
        storage_path = "/data";
    if (statvfs(storage_path, &vfs) != 0)
        storage_path = "/";
    if (statvfs(storage_path, &vfs) == 0) {
        device->storage_total = (size_t)vfs.f_frsize * vfs.f_blocks;
        device->storage_free = (size_t)vfs.f_frsize * vfs.f_bfree;
    }

    // Storage type
#if defined(__ANDROID__)
    {
        FILE* pf = popen("getprop ro.boot.bootdevice 2>/dev/null", "r");
        if (pf) {
            char name[128] = {0};
            if (fgets(name, sizeof(name), pf)) {
                if (strstr(name, "ufs")) device->storage_type = CALM_STORAGE_UFS;
                else if (strstr(name, "emmc") || strstr(name, "sdhc"))
                    device->storage_type = CALM_STORAGE_EMMC;
                else device->storage_type = CALM_STORAGE_UFS;
            }
            pclose(pf);
        } else {
            device->storage_type = CALM_STORAGE_UFS;
        }
    }
#elif defined(__linux__)
    if (access("/dev/nvme0", F_OK) == 0)
        device->storage_type = CALM_STORAGE_NVME;
    else
        device->storage_type = CALM_STORAGE_SSD;
#elif defined(__APPLE__)
    device->storage_type = CALM_STORAGE_NVME;
#endif

    return CALM_OK;
}

CalmError calm_device_check(const CalmDevice* device,
                                 size_t model_bytes,
                                 size_t context_length,
                                 size_t kv_bytes_per_token) {
    if (!device) return CALM_ERR_GENERIC;

    size_t total_needed = model_bytes + context_length * kv_bytes_per_token + ((size_t)512 << 20);
    size_t available = device->ram_available + device->swap_free;

    if (total_needed > available)
        return CALM_ERR_OOM;

    return CALM_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * 2. Quant Format Utilities
 * ═══════════════════════════════════════════════════════════════ */

const char* calm_quant_name(CalmQuantFormat q) {
    static const char* names[] = {
        "UNKNOWN", "F32", "F16",
        "Q8_0", "Q4_0", "Q4_1", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_K",
        "Q2_K", "Q3_XXS", "Q3_XS", "Q3_S",
        "IQ1_S", "IQ1_M",
        "IQ2_XXS", "IQ2_XS", "IQ2_S",
        "IQ3_XXS", "IQ3_XS", "IQ3_S",
        "IQ4_NL", "IQ4_XS",
        "TQ1_0", "BQ1_0",
    };
    if (q >= 0 && q < CALM_QUANT_COUNT) return names[q];
    return "UNKNOWN";
}

float calm_quant_bpw(CalmQuantFormat q) {
    switch (q) {
        case CALM_QUANT_F32: return 32.0f;
        case CALM_QUANT_F16: return 16.0f;
        case CALM_QUANT_Q8_0:
        case CALM_QUANT_Q8_K: return 8.0f;
        case CALM_QUANT_Q6_K: return 6.0f;
        case CALM_QUANT_Q5_K_M: return 5.0f;
        case CALM_QUANT_Q4_0:
        case CALM_QUANT_Q4_1:
        case CALM_QUANT_Q4_K_M: return 4.5f;
        case CALM_QUANT_Q3_XXS:
        case CALM_QUANT_IQ3_XS:
        case CALM_QUANT_IQ3_S: return 3.5f;
        case CALM_QUANT_Q2_K:
        case CALM_QUANT_IQ2_XXS:
        case CALM_QUANT_IQ2_XS:
        case CALM_QUANT_IQ2_S: return 2.5f;
        case CALM_QUANT_IQ1_S:
        case CALM_QUANT_IQ1_M: return 1.5f;
        case CALM_QUANT_TQ1_0: return 1.58f;
        case CALM_QUANT_BQ1_0: return 1.125f;
        default: return 4.5f;
    }
}

static CalmQuantFormat quant_from_name(const char* name) {
    if (!name) return CALM_QUANT_UNKNOWN;
    // Нормализуем: uppercase
    char upper[256];
    int i;
    for (i = 0; name[i] && i < (int)sizeof(upper) - 1; i++) upper[i] = (char)toupper(name[i]);
    upper[i] = '\0';

    // Убираем точки и подчёркивания для сравнения
    for (i = 0; upper[i]; i++)
        if (upper[i] == '.' || upper[i] == '_' || upper[i] == '-')
            upper[i] = ' ';

    struct { const char* key; CalmQuantFormat val; } map[] = {
        {"F32", CALM_QUANT_F32},
        {"F16", CALM_QUANT_F16},
        {"BF16", CALM_QUANT_F16},
        {"Q8 0", CALM_QUANT_Q8_0},
        {"Q8 K", CALM_QUANT_Q8_K},
        {"Q6 K", CALM_QUANT_Q6_K},
        {"Q5 K M", CALM_QUANT_Q5_K_M},
        {"Q5 K", CALM_QUANT_Q5_K_M},
        {"Q4 K M", CALM_QUANT_Q4_K_M},
        {"Q4 K", CALM_QUANT_Q4_K_M},
        {"Q4 0", CALM_QUANT_Q4_0},
        {"Q4 1", CALM_QUANT_Q4_1},
        {"Q3 XXS", CALM_QUANT_Q3_XXS},
        {"Q3 XS", CALM_QUANT_Q3_XS},
        {"Q3 S", CALM_QUANT_Q3_S},
        {"Q3 K", CALM_QUANT_Q3_XXS},
        {"Q2 K", CALM_QUANT_Q2_K},
        {"IQ1 S", CALM_QUANT_IQ1_S},
        {"IQ1 M", CALM_QUANT_IQ1_M},
        {"IQ2 XXS", CALM_QUANT_IQ2_XXS},
        {"IQ2 XS", CALM_QUANT_IQ2_XS},
        {"IQ2 S", CALM_QUANT_IQ2_S},
        {"IQ3 XXS", CALM_QUANT_IQ3_XXS},
        {"IQ3 XS", CALM_QUANT_IQ3_XS},
        {"IQ3 S", CALM_QUANT_IQ3_S},
        {"IQ4 NL", CALM_QUANT_IQ4_NL},
        {"IQ4 XS", CALM_QUANT_IQ4_XS},
        {"TQ1 0", CALM_QUANT_TQ1_0},
        {"BQ1 0", CALM_QUANT_BQ1_0},
        {NULL, CALM_QUANT_UNKNOWN},
    };
    for (int j = 0; map[j].key; j++) {
        if (strstr(upper, map[j].key))
            return map[j].val;
    }
    return CALM_QUANT_UNKNOWN;
}

/* ═══════════════════════════════════════════════════════════════
 * 3. GGUF Metadata Reader
 * ═══════════════════════════════════════════════════════════════ */

#define GGUF_MAGIC 0x46554747u

// GGUF value types
#define GGUF_TYPE_UINT8 0
#define GGUF_TYPE_INT8 1
#define GGUF_TYPE_UINT16 2
#define GGUF_TYPE_INT16 3
#define GGUF_TYPE_UINT32 4
#define GGUF_TYPE_INT32 5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_BOOL 7
#define GGUF_TYPE_STRING 8
#define GGUF_TYPE_ARRAY 9
#define GGUF_TYPE_UINT64 10
#define GGUF_TYPE_INT64 11
#define GGUF_TYPE_FLOAT64 12

typedef struct {
    char* key;
    int type;
    union {
        uint8_t  v_uint8;
        int8_t   v_int8;
        uint16_t v_uint16;
        int16_t  v_int16;
        uint32_t v_uint32;
        int32_t  v_int32;
        float    v_float32;
        bool     v_bool;
        uint64_t v_uint64;
        int64_t  v_int64;
        double   v_float64;
        char*    v_string;
    };
} GGUFMetadataEntry;

typedef struct {
    GGUFMetadataEntry* entries;
    int count;
    int capacity;
} GGUFMetadata;

static void metadata_init(GGUFMetadata* m) {
    m->entries = NULL;
    m->count = 0;
    m->capacity = 0;
}

static void metadata_add(GGUFMetadata* m, const char* key, int type) {
    if (m->count >= m->capacity) {
        int new_cap = m->capacity ? m->capacity * 2 : 64;
        GGUFMetadataEntry* e = realloc(m->entries, new_cap * sizeof(GGUFMetadataEntry));
        if (!e) return;
        m->entries = e;
        m->capacity = new_cap;
    }
    GGUFMetadataEntry* e = &m->entries[m->count++];
    e->key = strdup(key);
    e->type = type;
    memset(&e->v_uint8, 0, sizeof(e->v_uint8));
}

static void metadata_free(GGUFMetadata* m) {
    if (!m || !m->entries) return;
    for (int i = 0; i < m->count; i++) {
        free(m->entries[i].key);
        if (m->entries[i].type == GGUF_TYPE_STRING)
            free(m->entries[i].v_string);
    }
    free(m->entries);
    m->entries = NULL;
    m->count = 0;
    m->capacity = 0;
}

static const char* metadata_get_string(const GGUFMetadata* m, const char* key) {
    for (int i = 0; i < m->count; i++)
        if (strcmp(m->entries[i].key, key) == 0 && m->entries[i].type == GGUF_TYPE_STRING)
            return m->entries[i].v_string;
    return NULL;
}

static uint64_t metadata_get_uint(const GGUFMetadata* m, const char* key, uint64_t def) {
    for (int i = 0; i < m->count; i++) {
        if (strcmp(m->entries[i].key, key) != 0) continue;
        switch (m->entries[i].type) {
            case GGUF_TYPE_UINT8:  return m->entries[i].v_uint8;
            case GGUF_TYPE_UINT16: return m->entries[i].v_uint16;
            case GGUF_TYPE_UINT32: return m->entries[i].v_uint32;
            case GGUF_TYPE_UINT64: return m->entries[i].v_uint64;
            case GGUF_TYPE_INT32:  return (uint64_t)m->entries[i].v_int32;
            case GGUF_TYPE_INT64:  return (uint64_t)m->entries[i].v_int64;
        }
    }
    return def;
}

// Read GGUF metadata from file (no tensor weights, just header)
static CalmError read_gguf_metadata(const char* path, GGUFMetadata* meta) {
    FILE* f = fopen(path, "rb");
    if (!f) return CALM_ERR_FILE_NOT_FOUND;

    // Magic
    uint32_t magic;
    if (fread(&magic, 1, 4, f) != 4) { fclose(f); return CALM_ERR_INVALID_FORMAT; }
    if (magic != GGUF_MAGIC) {
        // Проверяем big-endian
        if (magic != __builtin_bswap32(GGUF_MAGIC)) { fclose(f); return CALM_ERR_INVALID_FORMAT; }
    }

    // Version
    uint32_t version;
    if (fread(&version, 1, 4, f) != 4) { fclose(f); return CALM_ERR_INVALID_FORMAT; }

    (void)version; // используем позже для v1 vs v2+

    uint64_t tensor_count = 0, metadata_count = 0;

    if (version == 1) {
        uint32_t tc, mc;
        if (fread(&tc, 1, 4, f) != 4 || fread(&mc, 1, 4, f) != 4) {
            fclose(f); return CALM_ERR_INVALID_FORMAT;
        }
        tensor_count = tc;
        metadata_count = mc;
    } else {
        if (fread(&tensor_count, 1, 8, f) != 8 || fread(&metadata_count, 1, 8, f) != 8) {
            fclose(f); return CALM_ERR_INVALID_FORMAT;
        }
    }

    metadata_init(meta);

    for (uint64_t i = 0; i < metadata_count; i++) {
        // Read key length — GGUF spec uses uint32(4B), but some tools write uint64(8B)
        // Detect by peeking: if byte after uint32 is 0x00 → uint64
        uint8_t key_len_raw[8];
        if (fread(key_len_raw, 1, 4, f) != 4) { metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT; }
        uint32_t key_len_32 = (uint32_t)key_len_raw[0] | ((uint32_t)key_len_raw[1] << 8) |
                              ((uint32_t)key_len_raw[2] << 16) | ((uint32_t)key_len_raw[3] << 24);

        uint64_t key_len;
        // Peek at next byte to detect uint64 vs uint32 key_length
        int c = fgetc(f);
        if (c == EOF) { metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT; }
        if (c == 0x00) {
            // Probably uint64: the byte after uint32 key_len is 0x00 (upper bytes)
            // Read 3 more bytes to complete uint64, then combine
            uint8_t upper[3];
            if (fread(upper, 1, 3, f) != 3) { metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT; }
            key_len = (uint64_t)key_len_32 | ((uint64_t)upper[0] << 32) |
                      ((uint64_t)upper[1] << 40) | ((uint64_t)upper[2] << 48);
        } else {
            // uint32: push the peeked byte back and use uint32 key_len
            ungetc(c, f);
            key_len = key_len_32;
        }
        if (key_len > 4096) { metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT; }

        char* key = malloc((size_t)key_len + 1);
        if (!key) { metadata_free(meta); fclose(f); return CALM_ERR_OOM; }
        if (fread(key, 1, (size_t)key_len, f) != (size_t)key_len) {
            free(key); metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT;
        }
        key[key_len] = '\0';

        // Read value type
        uint32_t val_type;
        if (fread(&val_type, 1, 4, f) != 4) {
            free(key); metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT;
        }

        metadata_add(meta, key, (int)val_type);
        GGUFMetadataEntry* entry = &meta->entries[meta->count - 1];
        free(key);

        // Read value
        bool read_ok = true;
        switch (val_type) {
            case GGUF_TYPE_UINT8:  read_ok = (fread(&entry->v_uint8, 1, 1, f) == 1); break;
            case GGUF_TYPE_INT8:   read_ok = (fread(&entry->v_int8, 1, 1, f) == 1); break;
            case GGUF_TYPE_UINT16: read_ok = (fread(&entry->v_uint16, 1, 2, f) == 2); break;
            case GGUF_TYPE_INT16:  read_ok = (fread(&entry->v_int16, 1, 2, f) == 2); break;
            case GGUF_TYPE_UINT32: read_ok = (fread(&entry->v_uint32, 1, 4, f) == 4); break;
            case GGUF_TYPE_INT32:  read_ok = (fread(&entry->v_int32, 1, 4, f) == 4); break;
            case GGUF_TYPE_FLOAT32: read_ok = (fread(&entry->v_float32, 1, 4, f) == 4); break;
            case GGUF_TYPE_UINT64: read_ok = (fread(&entry->v_uint64, 1, 8, f) == 8); break;
            case GGUF_TYPE_INT64:  read_ok = (fread(&entry->v_int64, 1, 8, f) == 8); break;
            case GGUF_TYPE_FLOAT64: read_ok = (fread(&entry->v_float64, 1, 8, f) == 8); break;
            case GGUF_TYPE_BOOL: {
                uint8_t b;
                read_ok = (fread(&b, 1, 1, f) == 1);
                if (read_ok) entry->v_bool = (bool)b;
                break;
            }
            case GGUF_TYPE_STRING: {
                uint64_t s_len;
                read_ok = (fread(&s_len, 1, 8, f) == 8);
                if (read_ok) {
                    if (s_len > 65536) { s_len = 65536; }
                    entry->v_string = calloc(s_len + 1, 1);
                    if (entry->v_string) {
                        read_ok = (fread(entry->v_string, 1, s_len, f) == s_len);
                    } else {
                        read_ok = false;
                    }
                }
                break;
            }
            case GGUF_TYPE_ARRAY: {
                uint32_t arr_type;
                uint64_t arr_len;
                if (fread(&arr_type, 1, 4, f) != 4 ||
                    fread(&arr_len, 1, 8, f) != 8) {
                    metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT;
                }
                // Skip array elements (we don't need them for analysis)
                for (uint64_t j = 0; j < arr_len; j++) {
                    switch (arr_type) {
                        case GGUF_TYPE_UINT8:  fseek(f, 1, SEEK_CUR); break;
                        case GGUF_TYPE_INT8:   fseek(f, 1, SEEK_CUR); break;
                        case GGUF_TYPE_UINT16: fseek(f, 2, SEEK_CUR); break;
                        case GGUF_TYPE_INT16:  fseek(f, 2, SEEK_CUR); break;
                        case GGUF_TYPE_UINT32: fseek(f, 4, SEEK_CUR); break;
                        case GGUF_TYPE_INT32:  fseek(f, 4, SEEK_CUR); break;
                        case GGUF_TYPE_FLOAT32: fseek(f, 4, SEEK_CUR); break;
                        case GGUF_TYPE_UINT64: fseek(f, 8, SEEK_CUR); break;
                        case GGUF_TYPE_INT64:  fseek(f, 8, SEEK_CUR); break;
                        case GGUF_TYPE_FLOAT64: fseek(f, 8, SEEK_CUR); break;
                        case GGUF_TYPE_BOOL:   fseek(f, 1, SEEK_CUR); break;
                        case GGUF_TYPE_STRING: {
                            // String: uint64 length + data
                            uint64_t elem_len;
                            if (fread(&elem_len, 1, 8, f) != 8) {
                                metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT;
                            }
                            if (elem_len > 0)
                                fseek(f, (long)elem_len, SEEK_CUR);
                            break;
                        }
                        default: fseek(f, 4, SEEK_CUR); break;
                    }
                }
                break;
            }
            default:
                // Unknown type — skip 4 bytes
                fseek(f, 4, SEEK_CUR);
                break;
        }
        if (!read_ok) {
            metadata_free(meta); fclose(f); return CALM_ERR_INVALID_FORMAT;
        }
    }

    fclose(f);
    return CALM_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * 4. Model Analysis
 * ═══════════════════════════════════════════════════════════════ */

static bool is_moe_architecture(const char* arch) {
    if (!arch) return false;
    const char* moe_keywords[] = {
        "mixtral", "deepseek", "moe", "dbrx", "jamba",
        "qwen2moe", "glm", "qwen3moe", NULL
    };
    for (int i = 0; moe_keywords[i]; i++)
        if (strstr(arch, moe_keywords[i]))
            return true;
    return false;
}

CalmError calm_model_read_info(const char* path, CalmModelInfo* info) {
    if (!path || !info) return CALM_ERR_GENERIC;
    memset(info, 0, sizeof(CalmModelInfo));

    // File size
    struct stat st;
    if (stat(path, &st) != 0) return CALM_ERR_FILE_NOT_FOUND;
    info->file_size = (size_t)st.st_size;

    // Default name from filename
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;
    strncpy(info->name, base, sizeof(info->name) - 1);

    // Read GGUF metadata
    GGUFMetadata meta;
    CalmError err = read_gguf_metadata(path, &meta);
    if (err != CALM_OK) return err;

    // Architecture
    const char* arch = metadata_get_string(&meta, "general.architecture");
    if (arch) {
        strncpy(info->arch_name, arch, sizeof(info->arch_name) - 1);

        // Определяем тип архитектуры
        if (strcmp(arch, "llama") == 0 || strcmp(arch, "qwen2") == 0 ||
            strcmp(arch, "qwen3") == 0 || strcmp(arch, "qwen35") == 0 ||
            strcmp(arch, "mistral") == 0 || strcmp(arch, "falcon") == 0 ||
            strcmp(arch, "gemma") == 0 || strcmp(arch, "phi3") == 0 ||
            strcmp(arch, "starcoder") == 0 || strcmp(arch, "olmo") == 0 ||
            strcmp(arch, "command-r") == 0) {
            info->arch_type = CALM_ARCH_DENSE;
        } else if (is_moe_architecture(arch)) {
            info->arch_type = CALM_ARCH_MOE;
        } else if (strcmp(arch, "bitnet") == 0) {
            info->arch_type = CALM_ARCH_BITNET;
        } else {
            info->arch_type = CALM_ARCH_DENSE; // default
        }
    }

    // Build metadata key prefix from architecture name
    char meta_prefix[64] = "llama.";
    if (arch && arch[0]) {
        snprintf(meta_prefix, sizeof(meta_prefix), "%s.", arch);
    }

    // GGUF version
    info->gguf_version = (uint32_t)metadata_get_uint(&meta, "general.gguf_version", 3);

    // Parameters
    info->parameter_count = metadata_get_uint(&meta, "general.parameter_count", 0);

    // Если параметров нет — оцениваем из имени
    if (info->parameter_count == 0) {
        const char* stem = base;
        const char* dot = strrchr(stem, '.');
        size_t stem_len = dot ? (size_t)(dot - stem) : strlen(stem);

        // Ищем "N.B" или "NB" в имени файла
        char name_copy[256];
        snprintf(name_copy, sizeof(name_copy), "%.*s", (int)stem_len, stem);

        // Разбиваем по разделителям
        char* token = strtok(name_copy, "-_.");
        while (token) {
            size_t tl = strlen(token);
            if (tl >= 2) {
                char suffix = toupper(token[tl - 1]);
                if (suffix == 'B' || suffix == 'M') {
                    char num_part[64];
                    strncpy(num_part, token, tl - 1);
                    num_part[tl - 1] = '\0';
                    double val = atof(num_part);
                    if (val > 0) {
                        info->parameter_count = (uint64_t)(val * (suffix == 'B' ? 1e9 : 1e6));
                        if (suffix == 'B') break;
                    }
                }
            }
            token = strtok(NULL, "-_.");
        }
    }

    // Helper: look up metadata with architecture prefix + fallback to "llama."
    // This handles both "llama.block_count" (older) and "qwen35.block_count" (newer) etc.
    #define META_GET(field, def_val) ({ \
        char _k1[128]; \
        snprintf(_k1, sizeof(_k1), "%s" field, meta_prefix); \
        uint64_t _v = metadata_get_uint(&meta, _k1, UINT64_MAX); \
        if (_v == UINT64_MAX) { \
            char _k2[128]; \
            snprintf(_k2, sizeof(_k2), "llama." field); \
            _v = metadata_get_uint(&meta, _k2, (uint64_t)(def_val)); \
        } \
        _v; \
    })

    // Block count, context, embedding (architecture-aware)
    info->block_count = (uint32_t)META_GET("block_count", 0);
    info->context_length = (uint32_t)META_GET("context_length", 0);
    info->embedding_length = (uint32_t)META_GET("embedding_length", 0);

    // MoE detection (architecture-aware)
    info->expert_count = (uint32_t)META_GET("expert_count", 0);
    info->expert_active = (uint32_t)META_GET("expert_used_count", 0);

    #undef META_GET

    if (info->expert_count > 0) {
        info->is_moe = true;
        info->arch_type = CALM_ARCH_MOE;
    }

    if (info->arch_type == CALM_ARCH_MOE && info->expert_count == 0)
        info->expert_count = 8;
    if (info->arch_type == CALM_ARCH_MOE && info->expert_active == 0)
        info->expert_active = 2;

    // Quant format
    int file_type = (int)metadata_get_uint(&meta, "general.file_type", 0xFFFFFFFF);
    if (file_type != 0xFFFFFFFF) {
        static const int ft_to_quant[] = {
            [0] = CALM_QUANT_F32,
            [1] = CALM_QUANT_F16,
            [2] = CALM_QUANT_Q4_0,
            [3] = CALM_QUANT_Q4_1,
            [4] = CALM_QUANT_Q5_K_M,  // Q5_0 → K_M
            [5] = CALM_QUANT_Q5_K_M,  // Q5_1 → K_M
            [6] = CALM_QUANT_Q8_0,
            [7] = CALM_QUANT_Q8_0,    // Q8_1
            [10] = CALM_QUANT_Q2_K,
            [11] = CALM_QUANT_Q3_XXS,
            [12] = CALM_QUANT_Q4_K_M,
            [13] = CALM_QUANT_Q5_K_M,
            [14] = CALM_QUANT_Q6_K,
            [15] = CALM_QUANT_Q8_K,
            [16] = CALM_QUANT_IQ1_S,
            [17] = CALM_QUANT_IQ1_M,
            [18] = CALM_QUANT_IQ2_XXS,
            [19] = CALM_QUANT_IQ2_XS,
            [20] = CALM_QUANT_IQ2_S,
            [26] = CALM_QUANT_IQ4_NL,
            [27] = CALM_QUANT_IQ4_XS,
        };
        if (file_type >= 0 && file_type < 28)
            info->quant = ft_to_quant[file_type];
    }

    // Fallback: detect from filename
    if (info->quant == CALM_QUANT_UNKNOWN) {
        info->quant = quant_from_name(base);
    }

    // Active parameters (for MoE)
    info->active_parameters = info->parameter_count;
    if (info->is_moe && info->expert_count > 0) {
        // Оценка: dense ≈ 1/3 параметров, эксперты ≈ 2/3
        uint64_t dense_params = info->parameter_count / 3;
        uint64_t expert_total = info->parameter_count - dense_params;
        uint64_t per_expert = info->expert_count > 0 ? expert_total / info->expert_count : 0;
        info->active_parameters = dense_params + per_expert * info->expert_active;
    }

    // Memory estimates
    float bpw = calm_quant_bpw(info->quant);
    info->weights_ram = (size_t)(info->parameter_count * bpw / 8);
    if (info->is_moe) {
        // Для MoE: плотная часть в RAM, эксперты могут стримиться
        // Но weights_ram указывает полный размер весов
    }

    // KV cache: ~2 bytes per element, per layer, per token
    // KV = 2 * n_layers * hidden_dim * 2 * sizeof(float16) per token
    uint64_t kv_elements = (uint64_t)info->block_count * info->embedding_length * 2;
    info->kv_per_token = kv_elements * 2; // fp16 = 2 bytes

    if (info->kv_per_token == 0)
        info->kv_per_token = 512 * 1024; // ~512 KB default

    metadata_free(&meta);

    return CALM_OK;
}

void calm_model_info_free(CalmModelInfo* info) {
    (void)info;
    // Currently no dynamic allocations in CalmModelInfo itself
}

/* ═══════════════════════════════════════════════════════════════
 * 5. Strategy / Planner
 * ═══════════════════════════════════════════════════════════════ */

const char* calm_strategy_name(CalmStrategy s) {
    switch (s) {
        case CALM_STRATEGY_FULL_LOAD: return "Full load (all in RAM)";
        case CALM_STRATEGY_MMAP: return "Memory-mapped (mmap)";
        case CALM_STRATEGY_MOE_STREAM: return "MoE streaming (Colibri-style)";
        case CALM_STRATEGY_LAYER_OFFLOAD: return "Layer offload (CPU+GPU)";
        case CALM_STRATEGY_HYBRID_TIER: return "Hybrid tier (hot/cold)";
        default: return "Unknown";
    }
}

CalmError calm_plan_create(const CalmDevice* device,
                                const CalmModelInfo* model,
                                CalmPlan* plan) {
    if (!device || !model || !plan) return CALM_ERR_GENERIC;
    memset(plan, 0, sizeof(CalmPlan));

    plan->cpu_threads = device->cpu_cores_logical > 1 ? device->cpu_cores_logical - 1 : 1;
    plan->context_length = 2048;
    plan->gpu_layers = -1; // auto
    plan->use_mmap = true;
    plan->kv_quant = CALM_QUANT_F16;
    plan->target_quant = CALM_QUANT_UNKNOWN; // keep

    size_t ram_avail = device->ram_available;

    // Strategy
    size_t weights_ram = model->weights_ram;
    if (model->is_moe) {
        // Для MoE: плотная часть + кэш экспертов
        weights_ram = weights_ram / 3 + ((size_t)512 << 20); // dense + 512 MB cache
    }

    if (weights_ram < ram_avail * 0.75) {
        plan->strategy = CALM_STRATEGY_FULL_LOAD;
        plan->use_mmap = false;
    } else if (model->is_moe && weights_ram > ram_avail) {
        plan->strategy = CALM_STRATEGY_MOE_STREAM;
        plan->use_mmap = true;
        plan->expert_cache_bytes = (ram_avail - ((size_t)512 << 20)) / 2;
        if (plan->expert_cache_bytes < (size_t)256 << 20)
            plan->expert_cache_bytes = (size_t)256 << 20;
        if (plan->expert_cache_bytes > (size_t)4 << 30)
            plan->expert_cache_bytes = (size_t)4 << 30;
    } else {
        plan->strategy = CALM_STRATEGY_MMAP;
        plan->use_mmap = true;
    }

    // GPU
    if (device->has_vulkan || device->has_metal || device->has_cuda || device->has_opencl) {
        plan->gpu_layers = 99; // max GPU
    } else {
        plan->gpu_layers = 0; // CPU only
    }

    // Context length
    size_t kv_for_4k = model->kv_per_token * 4096;
    if (kv_for_4k < ram_avail / 5) {
        plan->context_length = 8192;
    } else if (kv_for_4k < ram_avail * 2 / 5) {
        plan->context_length = 4096;
    } else {
        plan->context_length = 2048;
    }

    // Re-quant recommendation
    if (model->weights_ram > ram_avail * 3 / 2) {
        plan->target_quant = CALM_QUANT_BQ1_0; // 1-bit
    } else if (model->weights_ram > ram_avail * 6 / 5) {
        plan->target_quant = CALM_QUANT_TQ1_0; // ternary 1.58-bit
    }

    // Backend — auto-detect
    plan->backend = CALM_BACKEND_AUTO;

    return CALM_OK;
}

/* ═══════════════════════════════════════════════════════════════
 * 6. Backend Abstraction
 * ═══════════════════════════════════════════════════════════════ */

uint32_t calm_available_backends(void) {
    uint32_t mask = 0;
    // Native всегда "доступен" (но может не иметь реализации)
    mask |= (uint32_t)CALM_BACKEND_CPU;

    // Ollama
    if (system("ollama --version >/dev/null 2>&1") == 0)
        mask |= (1u << 4); // CALM_BACKEND_OLLAMA

    // llama.cpp
    if (system("llama-cli --version >/dev/null 2>&1") == 0 ||
        system("llama-server --version >/dev/null 2>&1") == 0)
        mask |= (1u << 5); // CALM_BACKEND_LLAMACPP

    return mask;
}

const char* calm_backend_name(CalmBackendType backend) {
    switch (backend) {
        case CALM_BACKEND_AUTO: return "auto";
        case CALM_BACKEND_CPU: return "cpu";
        case CALM_BACKEND_VULKAN: return "vulkan";
        case CALM_BACKEND_METAL: return "metal";
        case CALM_BACKEND_CUDA: return "cuda";
        case CALM_BACKEND_OLLAMA: return "ollama";
        case CALM_BACKEND_LLAMACPP: return "llamacpp";
        default: return "unknown";
    }
}

bool calm_has_vulkan(void) {
    return (access("/system/lib64/libvulkan.so", F_OK) == 0) ||
           (access("/vendor/lib64/libvulkan.so", F_OK) == 0) ||
           (system("vulkaninfo --version >/dev/null 2>&1") == 0);
}

bool calm_has_metal(void) {
#if defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

bool calm_has_cuda(void) {
    return (system("nvcc --version >/dev/null 2>&1") == 0) ||
           (system("nvidia-smi >/dev/null 2>&1") == 0);
}

int calm_optimal_gpu_layers(int max_layer_count,
                               size_t vram_bytes,
                               size_t layer_size_bytes) {
    if (vram_bytes == 0 || layer_size_bytes == 0) return 0;
    int layers = (int)(vram_bytes / layer_size_bytes);
    if (layers > max_layer_count) layers = max_layer_count;
    if (layers < 0) layers = 0;
    return layers;
}

/* ═══════════════════════════════════════════════════════════════
 * 7. Expert Cache
 * ═══════════════════════════════════════════════════════════════ */

CalmExpertCache* calm_cache_create(int capacity, int lookahead) {
    if (capacity <= 0) return NULL;
    CalmExpertCache* cache = calloc(1, sizeof(CalmExpertCache));
    if (!cache) return NULL;
    cache->slots = calloc((size_t)capacity, sizeof(CalmExpertSlot));
    if (!cache->slots) { free(cache); return NULL; }
    cache->capacity = capacity;
    cache->count = 0;
    cache->lookahead_distance = lookahead > 0 ? lookahead : 1;
    return cache;
}

CalmExpertSlot* calm_cache_get(CalmExpertCache* cache, int expert_id) {
    if (!cache) return NULL;
    for (int i = 0; i < cache->count; i++) {
        if (cache->slots[i].expert_id == expert_id) {
            cache->slots[i].last_accessed = cache->total_accesses++;
            cache->slots[i].access_count++;
            cache->slots[i].temperature = (float)cache->slots[i].access_count /
                                          (float)(cache->total_accesses + 1);
            return &cache->slots[i];
        }
    }
    return NULL;
}

CalmError calm_cache_put(CalmExpertCache* cache,
                              int expert_id,
                              const void* weights,
                              size_t size,
                              CalmQuantFormat quant) {
    if (!cache || !weights) return CALM_ERR_GENERIC;

    // Если уже есть — просто обновляем
    for (int i = 0; i < cache->count; i++) {
        if (cache->slots[i].expert_id == expert_id) {
            if (cache->slots[i].size != size) {
                free(cache->slots[i].weights);
                cache->slots[i].weights = malloc(size);
                if (!cache->slots[i].weights) return CALM_ERR_OOM;
            }
            memcpy(cache->slots[i].weights, weights, size);
            cache->slots[i].size = size;
            cache->slots[i].quant = quant;
            cache->slots[i].last_accessed = cache->total_accesses++;
            return CALM_OK;
        }
    }

    // Если есть место — добавляем
    if (cache->count < cache->capacity) {
        int idx = cache->count++;
        cache->slots[idx].expert_id = expert_id;
        cache->slots[idx].weights = malloc(size);
        if (!cache->slots[idx].weights) return CALM_ERR_OOM;
        memcpy(cache->slots[idx].weights, weights, size);
        cache->slots[idx].size = size;
        cache->slots[idx].quant = quant;
        cache->slots[idx].temperature = 1.0f;
        cache->slots[idx].last_accessed = cache->total_accesses++;
        cache->slots[idx].access_count = 1;
        return CALM_OK;
    }

    // Вытесняем coldest
    int coldest = 0;
    float min_temp = cache->slots[0].temperature;
    for (int i = 1; i < cache->capacity; i++) {
        if (cache->slots[i].temperature < min_temp) {
            min_temp = cache->slots[i].temperature;
            coldest = i;
        }
    }

    free(cache->slots[coldest].weights);
    cache->slots[coldest].expert_id = expert_id;
    cache->slots[coldest].weights = malloc(size);
    if (!cache->slots[coldest].weights) return CALM_ERR_OOM;
    memcpy(cache->slots[coldest].weights, weights, size);
    cache->slots[coldest].size = size;
    cache->slots[coldest].quant = quant;
    cache->slots[coldest].temperature = 1.0f;
    cache->slots[coldest].last_accessed = cache->total_accesses++;
    cache->slots[coldest].access_count = 1;

    return CALM_OK;
}

int calm_cache_predict(CalmExpertCache* cache,
                          const float* router_logits,
                          int n_experts,
                          int top_k,
                          int* predicted_ids) {
    if (!cache || !router_logits || !predicted_ids) return 0;

    // Простейший lookahead: выбираем top_k экспертов с наибольшими logits
    // В реальности здесь будет предсказание следующего слоя
    typedef struct { int id; float val; } Score;
    Score scores[256];
    int n = n_experts > 256 ? 256 : n_experts;

    for (int i = 0; i < n; i++) {
        scores[i].id = i;
        scores[i].val = router_logits[i];
    }

    // Sort by value descending (simple bubble sort for small n)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (scores[j].val < scores[j + 1].val) {
                Score tmp = scores[j];
                scores[j] = scores[j + 1];
                scores[j + 1] = tmp;
            }
        }
    }

    int k = top_k < n ? top_k : n;
    for (int i = 0; i < k; i++)
        predicted_ids[i] = scores[i].id;

    return k;
}

void calm_cache_clear(CalmExpertCache* cache) {
    if (!cache) return;
    for (int i = 0; i < cache->count; i++) {
        free(cache->slots[i].weights);
        cache->slots[i].weights = NULL;
    }
    cache->count = 0;
}

void calm_cache_free(CalmExpertCache* cache) {
    if (!cache) return;
    calm_cache_clear(cache);
    free(cache->slots);
    free(cache);
}

/* ═══════════════════════════════════════════════════════════════
 * 8. Runtime
 * ═══════════════════════════════════════════════════════════════ */

struct CalmRuntime {
    CalmDevice device;
    bool initialized;
};

struct CalmModel {
    CalmRuntime* runtime;
    CalmModelInfo info;
    CalmPlan plan;
    char path[CALM_MAX_PATH];
    // Native inference (loaded lazily)
    ct_gguf_context* gguf;
    ct_infer_state* infer;
    ct_tokenizer* tokenizer;
    // Статистика
    double gen_start_time;
    int tokens_generated;
    int prompt_tokens;           // last prompt token count (for API response)
};

CalmRuntime* calm_init(const CalmDevice* device) {
    CalmRuntime* rt = calloc(1, sizeof(CalmRuntime));
    if (!rt) return NULL;

    if (device) {
        rt->device = *device;
    } else {
        calm_device_probe(&rt->device);
    }

    rt->initialized = true;
    return rt;
}

CalmModel* calm_model_load(CalmRuntime* rt,
                                const char* path,
                                const CalmPlan* plan) {
    if (!rt || !path) return NULL;

    CalmModel* model = calloc(1, sizeof(CalmModel));
    if (!model) return NULL;

    model->runtime = rt;
    strncpy(model->path, path, sizeof(model->path) - 1);

    // Read model info
    CalmError err = calm_model_read_info(path, &model->info);
    if (err != CALM_OK) {
        fprintf(stderr, "Error reading model: %s\n", calm_err_string(err));
        free(model);
        return NULL;
    }

    // Create plan
    if (plan) {
        model->plan = *plan;
    } else {
        calm_plan_create(&rt->device, &model->info, &model->plan);
    }

    // Try to open GGUF for native inference
    model->gguf = ct_gguf_open(path);
    if (model->gguf) {
        model->infer = ct_infer_create(model->gguf);
        if (!model->infer) {
            // Not all architectures are supported — fall back to external backends
            ct_gguf_close(model->gguf);
            model->gguf = NULL;
        } else {
            fprintf(stderr, "✓ Native inference engine ready\n");
            // Load tokenizer from GGUF metadata
            model->tokenizer = ct_tokenizer_load(model->gguf);
            if (model->tokenizer)
                fprintf(stderr, "✓ Tokenizer loaded (%d tokens)\n",
                        model->tokenizer->vocab_size);
            else
                fprintf(stderr, "  (no tokenizer — will output token IDs)\n");
        }
    }

    return model;
}

// Forward declarations for backend implementations
CalmError calm_model_generate_native(CalmModel* model,
                                          const char* prompt,
                                          char* output,
                                          size_t output_size,
                                          const CalmGenerateParams* params);
CalmError calm_model_generate_ollama(CalmModel* model,
                                          const char* prompt,
                                          char* output,
                                          size_t output_size,
                                          const CalmGenerateParams* params);
CalmError calm_model_generate_llamacpp(CalmModel* model,
                                            const char* prompt,
                                            char* output,
                                            size_t output_size,
                                            const CalmGenerateParams* params);

CalmError calm_model_generate(CalmModel* model,
                                   const char* prompt,
                                   char* output,
                                   size_t output_size,
                                   const CalmGenerateParams* params) {
    if (!model || !prompt || !output || output_size == 0)
        return CALM_ERR_GENERIC;

    const char* backend = "auto";

    // Phase 1: native inference engine
    if (model->infer) {
        return calm_model_generate_native(model, prompt, output, output_size, params);
    }

    // Phase 0 fallback: делегируем внешним бэкендам
    int has_ollama = (system("ollama --version >/dev/null 2>&1") == 0);
    int has_llamacpp = (system("llama-cli --version >/dev/null 2>&1") == 0) ||
                       (access("./llama.cpp/build/bin/llama-cli", F_OK) == 0);

    if (has_ollama) {
        return calm_model_generate_ollama(model, prompt, output, output_size, params);
    } else if (has_llamacpp) {
        return calm_model_generate_llamacpp(model, prompt, output, output_size, params);
    } else {
        // Нет бэкенда — возвращаем информативное сообщение
        snprintf(output, output_size,
            "Calm: No inference backend found.\n"
            "  The model format may be unsupported, or no external backend is installed.\n"
            "Install Ollama (recommended):\n"
            "  curl -fsSL https://ollama.com/install.sh | sh\n\n"
            "Or build llama.cpp:\n"
            "  git clone --depth=1 https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp && cmake -B build && cmake --build build -j4\n");
        return CALM_ERR_BACKEND_UNAVAIL;
    }
}

// Backend: Ollama (через subprocess)
CalmError calm_model_generate_ollama(CalmModel* model,
                                          const char* prompt,
                                          char* output,
                                          size_t output_size,
                                          const CalmGenerateParams* params) {
    // Создаём временный Modelfile
    char mf_path[256] = "/tmp/calm_modelfile_XXXXXX";
    int fd = mkstemp(mf_path);
    if (fd < 0) {
        strncpy(mf_path, "/data/local/tmp/calm_mf_XXXXXX", sizeof(mf_path));
        fd = mkstemp(mf_path);
    }
    if (fd < 0) return CALM_ERR_GENERIC;

    FILE* mf = fdopen(fd, "w");
    if (!mf) { close(fd); return CALM_ERR_GENERIC; }
    fprintf(mf, "FROM %s\n", model->path);
    fprintf(mf, "PARAMETER num_ctx %u\n", model->plan.context_length);
    fclose(mf);

    // Имя модели в Ollama
    char model_name[256];
    const char* base = strrchr(model->path, '/');
    base = base ? base + 1 : model->path;
    snprintf(model_name, sizeof(model_name), "calm-%.200s", base);
    // Убираем расширение
    char* dot = strrchr(model_name, '.');
    if (dot) *dot = '\0';

    // ollama create
    char cmd[CALM_MAX_CMD];
    snprintf(cmd, sizeof(cmd),
             "ollama create '%.240s' -f '%.240s' 2>/dev/null", model_name, mf_path);
    int rc = system(cmd);

    unlink(mf_path);

    if (rc != 0) {
        snprintf(output, output_size,
                 "Calm: Failed to create Ollama model '%s'.\n", model_name);
        return CALM_ERR_BACKEND_UNAVAIL;
    }

    // Запускаем с промптом
    // Экранируем промпт для shell
    char escaped_prompt[CALM_MAX_STRING];
    size_t j = 0;
    for (size_t i = 0; prompt[i] && j < sizeof(escaped_prompt) - 4; i++) {
        if (prompt[i] == '\'') {
            escaped_prompt[j++] = '\'';
            escaped_prompt[j++] = '\\';
            escaped_prompt[j++] = '\'';
            escaped_prompt[j++] = '\'';
        } else {
            escaped_prompt[j++] = prompt[i];
        }
    }
    escaped_prompt[j] = '\0';

    // Собираем вывод
    snprintf(cmd, sizeof(cmd),
             "ollama run '%.240s' '%.4000s' 2>/dev/null", model_name, escaped_prompt);

    FILE* fp = popen(cmd, "r");
    if (!fp) {
        // Чистим модель
        char rm_cmd[512];
        snprintf(rm_cmd, sizeof(rm_cmd), "ollama rm '%.240s' >/dev/null 2>&1", model_name);
        system(rm_cmd);
        return CALM_ERR_BACKEND_UNAVAIL;
    }

    size_t total = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp) && total < output_size - 1) {
        size_t len = strlen(line);
        if (total + len >= output_size - 1)
            len = output_size - 1 - total;
        memcpy(output + total, line, len);
        total += len;
    }
    output[total] = '\0';

    int status = pclose(fp);

    // Чистим модель
    char rm_cmd[512];
    snprintf(rm_cmd, sizeof(rm_cmd), "ollama rm '%.240s' >/dev/null 2>&1", model_name);
    system(rm_cmd);

    return status == 0 ? CALM_OK : CALM_ERR_GENERIC;
}

// Backend: llama.cpp (через subprocess)
CalmError calm_model_generate_llamacpp(CalmModel* model,
                                            const char* prompt,
                                            char* output,
                                            size_t output_size,
                                            const CalmGenerateParams* params) {
    // Ищем llama-cli
    const char* search_paths[] = {
        "llama-cli",
        "llama-server",
        "./llama.cpp/build/bin/llama-cli",
        "./llama.cpp/build/bin/llama-server",
        "llama.cpp/build/bin/llama-cli",
        NULL
    };

    const char* llama_bin = NULL;
    for (int i = 0; search_paths[i]; i++) {
        if (system(search_paths[i]) == 0 || access(search_paths[i], X_OK) == 0) {
            // Проверяем, что это не "команда не найдена"
            // Для simplify используем access
            if (access(search_paths[i], X_OK) == 0 || strchr(search_paths[i], '/') == NULL) {
                llama_bin = search_paths[i];
                break;
            }
        }
        if (access(search_paths[i], F_OK) == 0) {
            llama_bin = search_paths[i];
            break;
        }
    }

    if (!llama_bin) {
        snprintf(output, output_size, "Calm: llama-cli not found.\n");
        return CALM_ERR_BACKEND_UNAVAIL;
    }

    char cmd[CALM_MAX_CMD];
    int n = snprintf(cmd, sizeof(cmd),
             "%s -m '%s' -c %u -t %d -p '",
             llama_bin, model->path,
             model->plan.context_length,
             model->plan.cpu_threads);

    if (model->plan.gpu_layers > 0) {
        n += snprintf(cmd + n, sizeof(cmd) - (size_t)n,
                      " -ngl %d", model->plan.gpu_layers);
    }

    // Добавляем промпт (безопасно)
    n += snprintf(cmd + n, sizeof(cmd) - (size_t)n, "%s'", prompt);

    // Обрезаем, если слишком длинный
    if ((size_t)n >= sizeof(cmd) - 1) {
        // Слишком длинная команда
        snprintf(output, output_size, "Calm: Prompt too long for CLI.\n");
        return CALM_ERR_GENERIC;
    }

    FILE* fp = popen(cmd, "r");
    if (!fp) return CALM_ERR_GENERIC;

    size_t total = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp) && total < output_size - 1) {
        size_t len = strlen(line);
        if (total + len >= output_size - 1)
            len = output_size - 1 - total;
        memcpy(output + total, line, len);
        total += len;
    }
    output[total] = '\0';
    pclose(fp);

    return CALM_OK;
}

double calm_model_progress(const CalmModel* model) {
    if (!model) return 0;
    return model->tokens_generated > 0 ? 1.0 : 0.0;
}

float calm_model_speed(const CalmModel* model) {
    if (!model || model->tokens_generated == 0) return 0;
    double elapsed = (double)(clock() - model->gen_start_time) / CLOCKS_PER_SEC;
    if (elapsed <= 0) return 0;
    return (float)(model->tokens_generated / elapsed);
}

/* ── Native inference backend ── */

/* ─── Tool calling: strip <|tool_call|> and trailing JSON from output ─── */
static void strip_tool_calls(char* text) {
    char* tag = strstr(text, "<|tool_call|>");
    if (!tag) return;
    /* Truncate at the tag */
    *tag = '\0';
}

/* ─── Tool calling: build the full chat prompt with optional tools ─── */
static int build_chat_prompt(ct_tokenizer* tok, const char* user_prompt,
                              const CalmToolDefinitions* tools,
                              int* tokens, int max_tokens) {
    char chat_buf[16384];
    int pos = 0;

    /* System message */
    int n = snprintf(chat_buf + pos, sizeof(chat_buf) - pos,
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
        "You are a helpful assistant.");
    if (n > 0 && (size_t)n < sizeof(chat_buf) - pos) pos += (size_t)n;

    /* Append tool definitions if any */
    if (tools && tools->count > 0) {
        char tool_buf[8192];
        ct_tools_format_system(tools, tool_buf, sizeof(tool_buf));
        n = snprintf(chat_buf + pos, sizeof(chat_buf) - pos, "%s", tool_buf);
        if (n > 0 && (size_t)n < sizeof(chat_buf) - pos) pos += (size_t)n;
    }

    /* Close system, add user */
    n = snprintf(chat_buf + pos, sizeof(chat_buf) - pos,
        "<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
        user_prompt);
    if (n > 0 && (size_t)n < sizeof(chat_buf) - pos) pos += (size_t)n;

    chat_buf[pos] = '\0';

    int n_tokens = ct_tokenizer_encode(tok, chat_buf, tokens, max_tokens);
    if (n_tokens < 1) n_tokens = 1;
    if (n_tokens >= max_tokens) n_tokens = max_tokens - 1;

    fprintf(stderr, "[CHAT] constructed %d tokens for chat format%s\n",
            n_tokens, (tools && tools->count > 0) ? " (with tools)" : "");
    return n_tokens;
}

/* ─── Tool calling: execute one generation step ─── */

/* ─── Tool calling: execute tool calls and append results to buffer ─── */
static int execute_tool_round(CalmToolCall* calls, int n_calls,
                               char* buf, size_t buf_size) {
    size_t pos = 0;
    for (int i = 0; i < n_calls && pos < buf_size; i++) {
        CalmToolResult result;
        memset(&result, 0, sizeof(result));
        ct_tools_execute(&calls[i], &result);

        char step_buf[CT_TOOL_RESULT_MAX + 256];
        ct_tools_format_result(&calls[i], &result, step_buf, sizeof(step_buf));

        size_t len = strlen(step_buf);
        if (pos + len < buf_size) {
            memcpy(buf + pos, step_buf, len);
            pos += len;
        }
    }
    buf[pos] = '\0';
    return (int)pos;
}

/* ─── Native inference with tool calling support ─── */

/* Helper struct for streaming token decode */
struct stream_decode_ctx {
    ct_tokenizer* tok;
    void (*user_cb)(const char*, void*);
    void* user_arg;
};

/* Stream callback: ct_infer_generate calls this per token → decode to text → call user cb */
static void stream_token_to_text(int token, void* ctx) {
    struct stream_decode_ctx* sc = (struct stream_decode_ctx*)ctx;
    char buf[128];
    ct_tokenizer_decode_single(sc->tok, token, buf, sizeof(buf));
    if (sc->user_cb) sc->user_cb(buf, sc->user_arg);
}

CalmError calm_model_generate_native(CalmModel* model,
                                           const char* prompt,
                                           char* output,
                                           size_t output_size,
                                           const CalmGenerateParams* params) {
    ct_infer_state* s = model->infer;
    if (!s) return CALM_ERR_BACKEND_UNAVAIL;

    model->gen_start_time = (double)clock() / CLOCKS_PER_SEC;
    int max_tokens = params ? params->max_tokens : 64;
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 512) max_tokens = 512;  // increased for tool calling

    float temp = params ? params->temperature : 0.7f;
    float top_p_val = params ? params->top_p : 1.0f;
    float rp_val = params ? params->repeat_penalty : 1.0f;
    int top_k_val = params ? (int)params->top_k : 0;
    const CalmToolDefinitions* tools = params ? (const CalmToolDefinitions*)params->tools : NULL;
    int max_rounds = params ? params->max_tool_rounds : 5;
    if (max_rounds < 1) max_rounds = 1;
    if (max_rounds > 10) max_rounds = 10;

    if (!model->tokenizer || !prompt || !prompt[0]) {
        // No tokenizer: simple fallback
        int in_token = 1;
        if (prompt && prompt[0] >= '0' && prompt[0] <= '9')
            in_token = atoi(prompt);
        int tokens[64] = {in_token};
        int eos_id = -1;
        int total = ct_infer_generate(s, tokens, 1, max_tokens, temp, eos_id, tokens, NULL, NULL,
                                       1.0f, 1.0f, 0);
        if (total < 1) {
            snprintf(output, output_size, "Inference failed.\n");
            return CALM_ERR_GENERIC;
        }
        snprintf(output, output_size, "[Native — %d tokens, no tokenizer]\n", total - 1);
        return CALM_OK;
    }

    /* ── Build initial chat prompt ── */
    int tokens[4096];
    int n_prompt = build_chat_prompt(model->tokenizer, prompt, tools, tokens, 2048);
    model->prompt_tokens = n_prompt;
    int eos_id = model->tokenizer->eos_id;

    /* ── Tool calling loop ── */
    int has_tools = (tools && tools->count > 0);
    int round = 0;

    /* Setup streaming callback if requested (no tools = single round, safe to stream) */
    struct stream_decode_ctx stream_data;
    int should_stream = params && params->on_token && !has_tools;
    if (should_stream) {
        stream_data.tok = model->tokenizer;
        stream_data.user_cb = params->on_token;
        stream_data.user_arg = params->user_data;
    }

    /* Helper: re-encode full_history and generate */
    int total = ct_infer_generate(s, tokens, n_prompt, max_tokens, temp, eos_id, tokens,
                                   should_stream ? stream_token_to_text : NULL,
                                   should_stream ? &stream_data : NULL,
                                   top_p_val, rp_val, top_k_val);

    if (total < 1) {
        snprintf(output, output_size, "Inference failed.\n");
        return CALM_ERR_GENERIC;
    }
    model->tokens_generated = total - n_prompt;

    if (model->tokens_generated == 0) {
        snprintf(output, output_size, "No tokens generated.\n");
        return CALM_OK;
    }

    /* Decode the first response */
    char response[CALM_MAX_STRING];
    ct_tokenizer_decode(model->tokenizer,
                         tokens + n_prompt,
                         model->tokens_generated,
                         response, sizeof(response));

    if (!has_tools) {
        /* No tools — just return the decoded response */
        strncpy(output, response, output_size - 1);
        output[output_size - 1] = '\0';
        return CALM_OK;
    }

    /* ── Tool loop ── */
    /* We need to track the conversation across rounds.
     * Strategy: rebuild the full chat each round with all previous exchanges. */
    char conversation[CALM_MAX_STRING] = {0};
    size_t conv_pos = 0;

    {
        char init_buf[16384];
        int n = snprintf(init_buf, sizeof(init_buf),
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
            "You are a helpful assistant.");
        int p = n > 0 && (size_t)n < sizeof(init_buf) ? (size_t)n : 0;
        if (tools && tools->count > 0) {
            char tool_buf[8192];
            ct_tools_format_system(tools, tool_buf, sizeof(tool_buf));
            n = snprintf(init_buf + p, sizeof(init_buf) - (size_t)p, "%s", tool_buf);
            if (n > 0 && (size_t)n < sizeof(init_buf) - (size_t)p) p += (size_t)n;
        }
        n = snprintf(init_buf + p, sizeof(init_buf) - (size_t)p,
            "<|im_end|>\n");
        if (n > 0 && (size_t)n < sizeof(init_buf) - (size_t)p) p += (size_t)n;
        init_buf[p] = '\0';
        if ((size_t)p < sizeof(conversation)) {
            memcpy(conversation, init_buf, (size_t)p + 1);
            conv_pos = (size_t)p;
        }
    }

    /* Add user message */
    {
        int n = snprintf(conversation + conv_pos, sizeof(conversation) - conv_pos,
            "<|im_start|>user\n%s<|im_end|>\n", prompt);
        if (n > 0 && (size_t)n < sizeof(conversation) - conv_pos)
            conv_pos += (size_t)n;
    }

    /* Initial assistant response */
    {
        int n = snprintf(conversation + conv_pos, sizeof(conversation) - conv_pos,
            "<|im_start|>assistant\n%s", response);
        if (n > 0 && (size_t)n < sizeof(conversation) - conv_pos)
            conv_pos += (size_t)n;
    }

    /* Tool loop */
    for (round = 0; round < max_rounds; round++) {
        /* Check if the last assistant response contains tool calls */
        CalmToolCall calls[CT_TOOL_CALLS_MAX];
        int n_calls = ct_tools_parse(response, calls, CT_TOOL_CALLS_MAX);

        if (n_calls == 0) {
            /* No more tool calls — done */
            break;
        }

        fprintf(stderr, "[TOOLS] round %d: %d tool call(s)\n", round + 1, n_calls);

        /* Execute tool calls and build the tool response chunk */
        char tool_result[CT_TOOL_RESULT_MAX * 2];
        execute_tool_round(calls, n_calls, tool_result, sizeof(tool_result));

        /* Add tool results to conversation */
        int n = snprintf(conversation + conv_pos, sizeof(conversation) - conv_pos,
            "%s", tool_result);
        if (n > 0 && (size_t)n < sizeof(conversation) - conv_pos)
            conv_pos += (size_t)n;

        /* Encode full conversation and generate next step */
        n_prompt = ct_tokenizer_encode(model->tokenizer, conversation,
                                        tokens, 2048);
        if (n_prompt < 1) break;
        if (n_prompt >= 2048) n_prompt = 2048 - 1;

        total = ct_infer_generate(s, tokens, n_prompt, max_tokens, temp, eos_id, tokens, NULL, NULL,
                                   top_p_val, rp_val, top_k_val);
        if (total < 1) break;

        model->tokens_generated = total - n_prompt;
        if (model->tokens_generated <= 0) break;

        /* Decode next assistant response */
        ct_tokenizer_decode(model->tokenizer,
                             tokens + n_prompt,
                             model->tokens_generated,
                             response, sizeof(response));

        /* Append to conversation */
        n = snprintf(conversation + conv_pos, sizeof(conversation) - conv_pos,
            "%s", response);
        if (n > 0 && (size_t)n < sizeof(conversation) - conv_pos)
            conv_pos += (size_t)n;
    }

    /* ── Final output: strip tool calls from last response ── */
    strip_tool_calls(response);
    if (response[0] == '\0') {
        /* If the last response was only tool calls, use the response before that */
        /* Extract the text from conversation before the last assistant turn */
        strncpy(output, "Tool execution complete.", output_size - 1);
        output[output_size - 1] = '\0';
    } else {
        strncpy(output, response, output_size - 1);
        output[output_size - 1] = '\0';
    }

    if (round > 0) {
        fprintf(stderr, "[TOOLS] completed after %d round(s)\n", round);
    }

    return CALM_OK;
}

void calm_model_free(CalmModel* model) {
    if (!model) return;
    ct_tokenizer_free(model->tokenizer);
    ct_infer_free(model->infer);
    ct_gguf_close(model->gguf);
    free(model);
}

void calm_destroy(CalmRuntime* rt) {
    free(rt);
}

/* ═══════════════════════════════════════════════════════════════
 * 9. Display Helpers (CLI)
 * ═══════════════════════════════════════════════════════════════ */

// ANSI colors (only when stdout is a terminal)
#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_RED     "\033[31m"
#define ANSI_GRAY    "\033[90m"

static bool use_color(void) {
    return isatty(STDOUT_FILENO);
}

#define C(c) (use_color() ? (c) : "")

static void progress_bar(double pct, int width) {
    if (!use_color()) return;
    int filled = (int)(pct * width);
    printf(" ");
    for (int i = 0; i < width; i++)
        printf("%s", i < filled ? "█" : "░");
    printf(" ");
}

static void print_device_info(const CalmDevice* d) {
    char fmt_buf[64];

    printf("\n%s%s Calm Device Scan%s\n", C(ANSI_BOLD), C(ANSI_CYAN), C(ANSI_RESET));
    printf("───────────────────────────────────────────────────────\n");

    printf("\n%sPlatform%s\n", C(ANSI_BOLD), C(ANSI_RESET));
    printf("  OS: ");
    switch (d->platform) {
        case CALM_PLATFORM_ANDROID: printf("Android"); break;
        case CALM_PLATFORM_LINUX:   printf("Linux"); break;
        case CALM_PLATFORM_MACOS:   printf("macOS"); break;
        case CALM_PLATFORM_WINDOWS: printf("Windows"); break;
        default: printf("Unknown");
    }
    if (d->is_termux) printf(" (Termux)");
    printf("\n");

    printf("\n%sCPU%s\n", C(ANSI_BOLD), C(ANSI_RESET));
    printf("  Cores: %d logical / %d physical\n",
           d->cpu_cores_logical, d->cpu_cores_physical);
    if (d->cpu_max_freq_mhz > 0)
        printf("  Max freq: %d MHz\n", d->cpu_max_freq_mhz);

    printf("\n%sMemory%s\n", C(ANSI_BOLD), C(ANSI_RESET));
    double ram_pct = d->ram_total > 0 ?
        (double)(d->ram_total - d->ram_available) / d->ram_total : 0;
    printf("  RAM:");
    progress_bar(ram_pct, 20);
    calm_format_size(fmt_buf, sizeof(fmt_buf), d->ram_available);
    printf(" %s free / ", fmt_buf);
    calm_format_size(fmt_buf, sizeof(fmt_buf), d->ram_total);
    printf("%s total\n", fmt_buf);

    if (d->swap_total > 0) {
        double sw_pct = (double)(d->swap_total - d->swap_free) / d->swap_total;
        printf("  Swap:");
        progress_bar(sw_pct, 20);
        calm_format_size(fmt_buf, sizeof(fmt_buf), d->swap_free);
        printf(" %s free / ", fmt_buf);
        calm_format_size(fmt_buf, sizeof(fmt_buf), d->swap_total);
        printf("%s total\n", fmt_buf);
    }

    printf("\n%sStorage%s\n", C(ANSI_BOLD), C(ANSI_RESET));
    const char* st_names[] = {"Unknown", "NVMe", "UFS", "SSD", "eMMC"};
    printf("  Type: %s\n", st_names[d->storage_type]);
    double st_pct = d->storage_total > 0 ?
        (double)(d->storage_total - d->storage_free) / d->storage_total : 0;
    printf("  Space:");
    progress_bar(st_pct, 20);
    calm_format_size(fmt_buf, sizeof(fmt_buf), d->storage_free);
    printf(" %s free / ", fmt_buf);
    calm_format_size(fmt_buf, sizeof(fmt_buf), d->storage_total);
    printf("%s total\n", fmt_buf);

    printf("\n%sGPU%s\n", C(ANSI_BOLD), C(ANSI_RESET));
    printf("  Vulkan: %s\n", d->has_vulkan ? "✅" : "❌");
    printf("  OpenCL: %s\n", d->has_opencl ? "✅" : "❌");
    printf("  Metal:  %s\n", d->has_metal ? "✅" : "❌");
    printf("  CUDA:   %s\n", d->has_cuda ? "✅" : "❌");

    printf("\n%sBackends%s\n", C(ANSI_BOLD), C(ANSI_RESET));
    uint32_t backends = calm_available_backends();
    printf("  Native:   ✅\n");
    printf("  Ollama:   %s\n", (backends & (1u << 4)) ? "✅" : "❌");
    printf("  llama.cpp: %s\n", (backends & (1u << 5)) ? "✅" : "❌");

    printf("\n");
}

static void print_model_info(const CalmModelInfo* m, const CalmPlan* plan) {
    char fmt_buf[64];

    printf("\n%s%s Model Analysis: %s%s\n", C(ANSI_BOLD), C(ANSI_CYAN), m->name, C(ANSI_RESET));
    printf("───────────────────────────────────────────────────────\n");

    const char* arch_names[] = {"Unknown", "Dense", "MoE", "Hybrid", "BitNet"};
    printf("  Architecture:  %s", arch_names[m->arch_type]);
    if (m->arch_name[0]) printf(" (%s)", m->arch_name);
    printf("\n");

    char params_str[32];
    calm_format_params(params_str, sizeof(params_str), m->parameter_count);
    printf("  Parameters:    %s (%" PRIu64 ")\n", params_str, m->parameter_count);

    if (m->is_moe) {
        calm_format_params(params_str, sizeof(params_str), m->active_parameters);
        printf("  MoE:           %u experts, %u active/token (~%s active)\n",
               m->expert_count, m->expert_active, params_str);
    }

    printf("  Quant:         %s\n", calm_quant_name(m->quant));
    printf("  File size:     ");
    calm_format_size(fmt_buf, sizeof(fmt_buf), m->file_size);
    printf("%s\n", fmt_buf);
    printf("  Blocks:        %u\n", m->block_count);
    printf("  Context:       %u tokens\n", m->context_length);
    printf("  Embed dim:     %u\n", m->embedding_length);

    printf("\n  %sMemory estimate%s\n", C(ANSI_BOLD), C(ANSI_RESET));
    calm_format_size(fmt_buf, sizeof(fmt_buf), m->weights_ram);
    printf("  Weights RAM:   %s\n", fmt_buf);
    calm_format_size(fmt_buf, sizeof(fmt_buf), m->kv_per_token);
    printf("  KV cache:      %s/token\n", fmt_buf);

    size_t total_est = m->weights_ram + m->kv_per_token * 2048 + ((size_t)512 << 20);
    calm_format_size(fmt_buf, sizeof(fmt_buf), total_est);
    printf("  Total est.:    %s (@ 2048 ctx)\n", fmt_buf);

    if (plan) {
        printf("\n  %sRecommended plan%s\n", C(ANSI_BOLD), C(ANSI_RESET));
        printf("  Strategy:      %s\n", calm_strategy_name(plan->strategy));
        printf("  Backend:       %s\n", calm_backend_name(plan->backend));
        printf("  Context:       %u tokens\n", plan->context_length);
        printf("  GPU layers:    %s\n", plan->gpu_layers < 0 ? "auto" :
               plan->gpu_layers == 0 ? "CPU only" :
               plan->gpu_layers >= 99 ? "all" : "some");
        printf("  CPU cores:     %d\n", plan->cpu_threads);
        if (plan->target_quant != CALM_QUANT_UNKNOWN)
            printf("  Re-quant:      %s (recommended)\n",
                   calm_quant_name(plan->target_quant));
    }

    printf("\n");
}

static void print_performance_estimate(const CalmDevice* d,
                                        const CalmModelInfo* m,
                                        const CalmPlan* plan) {
    // Performance estimation (heuristic)
    double ram_avail_gb = d->ram_available / (double)(1 << 30);
    double params_b = m->parameter_count / 1e9;
    if (params_b <= 0) params_b = 1.0;

    double base_tok_s;
    if (d->platform == CALM_PLATFORM_ANDROID) {
        base_tok_s = 12.0 / pow(params_b, 0.6);
        if (m->weights_ram >= d->ram_available * 0.75)
            base_tok_s *= 0.3;
    } else {
        base_tok_s = 30.0 / pow(params_b, 0.5);
        if (m->weights_ram >= d->ram_available * 0.75)
            base_tok_s *= 0.4;
    }

    if (base_tok_s < 0.01) base_tok_s = 0.01;

    double with_vulkan = d->has_vulkan ? base_tok_s * 1.8 : base_tok_s;
    double with_metal = d->has_metal ? base_tok_s * 2.0 : with_vulkan;
    double best = with_vulkan > with_metal ? with_vulkan : with_metal;

    printf("\n%s%sPerformance Estimate%s\n", C(ANSI_BOLD), C(ANSI_CYAN), C(ANSI_RESET));
    printf("───────────────────────────────────────────────────────\n");

    printf("  CPU only:        %.1f tok/s\n", base_tok_s);
    if (d->has_vulkan)
        printf("  With Vulkan:     %.1f tok/s  (+%.0f%%)\n",
               with_vulkan, ((with_vulkan / base_tok_s) - 1) * 100);
    if (d->has_metal)
        printf("  With Metal:      %.1f tok/s  (+%.0f%%)\n",
               with_metal, ((with_metal / base_tok_s) - 1) * 100);

    if (m->is_moe) {
        printf("  MoE stream cold: %.1f tok/s\n", base_tok_s * 0.15);
        printf("  MoE stream warm: %.1f tok/s\n", base_tok_s * 0.5);
    }

    // TTFT
    double ttft = 4.0 / best;
    printf("\n  Estimated TTFT:  %.1f s\n", ttft);

    // Verdict
    const char* verdict;
    const char* color;
    if (best >= 10) {
        verdict = "Excellent — real-time interactive";
        color = C(ANSI_GREEN);
    } else if (best >= 3) {
        verdict = "Good — usable for chat";
        color = C(ANSI_GREEN);
    } else if (best >= 1) {
        verdict = "Okay — slow but usable";
        color = C(ANSI_YELLOW);
    } else if (best >= 0.1) {
        verdict = "Slow — batch processing only";
        color = C(ANSI_YELLOW);
    } else {
        verdict = "Very slow — may not be practical";
        color = C(ANSI_RED);
    }

    printf("\n  Verdict: %s%s%s\n", color, verdict, C(ANSI_RESET));
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════
 * 9b. HTTP Server (/v1/completions)
 * ═══════════════════════════════════════════════════════════════ */

/* Context passed to /v1/completions handler */
typedef struct {
    CalmModel* model;
    CalmToolDefinitions* tools;
    pthread_mutex_t mutex;
    char display_name[256]; /* optional display name, defaults to filename */
} serve_ctx;

/* Debug log: writes to /tmp/calm_debug.log with timestamp */
static void debug_log(const char* msg) {
    FILE* f = fopen("/data/data/com.termux/files/usr/tmp/calm_debug.log", "a");
    if (!f) return;
    time_t t = time(NULL);
    struct tm* tm = localtime(&t);
    fprintf(f, "[%02d:%02d:%02d] %s\n", tm->tm_hour, tm->tm_min, tm->tm_sec, msg);
    fclose(f);
}

/* Escape string for JSON (minimal — handles chars that break JSON) */
static void json_escape(const char* in, char* out, size_t out_size) {
    size_t j = 0;
    for (const char* p = in; *p && j < out_size - 1; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '"':  if (j + 2 < out_size) { out[j++] = '\\'; out[j++] = '"'; } break;
            case '\\': if (j + 2 < out_size) { out[j++] = '\\'; out[j++] = '\\'; } break;
            case '\n': if (j + 2 < out_size) { out[j++] = '\\'; out[j++] = 'n'; } break;
            case '\r': if (j + 2 < out_size) { out[j++] = '\\'; out[j++] = 'r'; } break;
            case '\t': if (j + 2 < out_size) { out[j++] = '\\'; out[j++] = 't'; } break;
            default:
                if (c < 0x20) {
                    if (j + 6 < out_size)
                        j += (size_t)snprintf(out + j, out_size - j, "\\u%04x", c);
                } else {
                    out[j++] = c;
                }
        }
    }
    out[j] = '\0';
}

/* ─── /v1/completions + /v1/chat/completions handler (OpenAI API compatible) ─── */

/* Format messages array into a ChatML prompt.
 * Returns pointer to static buffer, NULL on error.
 * For Qwen2.5, the format is:
 *   <|im_start|>system\n{msg}<|im_end|>\n
 *   <|im_start|>user\n{msg}<|im_end|>\n
 *   <|im_start|>assistant\n{msg}<|im_end|>\n
 *   ...
 *   <|im_start|>assistant\n
 */
static const char* format_chatml(const char* body, char* out, size_t out_size) {
    if (!body || !out || out_size < 16) return NULL;
    out[0] = '\0';

    /* Local whitespace skipper */
    #define SKIP_WS(p) do { while (*(p) && (unsigned char)*(p) <= ' ') (p)++; } while(0)

    /* Find the messages array */
    const char* arr = ct_json_get_value(body, "messages");
    if (!arr || *arr != '[') return NULL;
    arr++; /* skip [ */

    const char* p = arr;
    char role[64];
    char content[CALM_MAX_STRING];
    size_t written = 0;
    int found = 0;

    while (*p) {
        SKIP_WS(p);
        if (!*p || *p == ']') break;

        if (*p == ',') { p++; continue; }

        if (*p == '{') {
            /* Find end of this object */
            const char* obj_end = p;
            int obj_depth = 0;
            while (*obj_end) {
                if (*obj_end == '{') { obj_depth++; obj_end++; }
                else if (*obj_end == '}') { obj_depth--; obj_end++; if (obj_depth == 0) break; }
                else if (*obj_end == '"') {
                    obj_end++;
                    while (*obj_end) {
                        if (*obj_end == '\\') { if (obj_end[1]) obj_end += 2; else break; }
                        else if (*obj_end == '"') { obj_end++; break; }
                        else obj_end++;
                    }
                } else obj_end++;
            }

            /* Extract this object */
            size_t obj_len = (size_t)(obj_end - p);
            if (obj_len < 2) { p = obj_end; continue; }
            char obj_buf[8192];
            if (obj_len >= sizeof(obj_buf)) { p = obj_end; continue; }
            memcpy(obj_buf, p, obj_len);
            obj_buf[obj_len] = '\0';

            /* Get role */
            const char* rv = ct_json_get_string(obj_buf, "role");
            if (!rv) { p = obj_end; continue; }
            size_t rl = strlen(rv);
            if (rl >= sizeof(role)) rl = sizeof(role) - 1;
            memcpy(role, rv, rl);
            role[rl] = '\0';

            /* Get content (optional for assistant — the model response) */
            const char* cv = ct_json_get_string(obj_buf, "content");
            if (!cv) cv = "";
            size_t cl = strlen(cv);
            if (cl >= sizeof(content)) cl = sizeof(content) - 1;
            memcpy(content, cv, cl);
            content[cl] = '\0';

            /* Format: <|im_start|>role\ncontent<|im_end|>\n */
            int n = snprintf(out + written, out_size - written,
                "<|im_start|>%s\n%s<|im_end|>\n", role, content);
            if (n < 0 || (size_t)n >= out_size - written) break;
            written += (size_t)n;
            found = 1;

            p = obj_end;
            continue;
        }
        p++;
    }

    /* Append the final assistant turn header */
    if (found) {
        snprintf(out + written, out_size - written, "<|im_start|>assistant\n");
    }

    #undef SKIP_WS
    return out;
}

/* Forward declaration for SSE */
static void json_escape(const char* in, char* out, size_t out_size);
static int gen_completions(serve_ctx* ctx, const char* prompt,
                           char* response_body, size_t response_size,
                           int* out_status_code, const char** out_content_type,
                           int max_tokens, float temperature, float top_p,
                           float top_k, float repeat_penalty, int is_chat,
                           int json_mode);

static int handle_v1_completions(const char* path, const char* method,
                                  const char* body,
                                  char* response_body, size_t response_size,
                                  int* out_status_code,
                                  const char** out_content_type,
                                  void* user_data) {
    serve_ctx* ctx = (serve_ctx*)user_data;

    debug_log("handle_v1_completions enter");

    /* /v1/models — list available models */
    if (strcmp(path, "/v1/models") == 0) {
        *out_status_code = 200;
        *out_content_type = "application/json";
        snprintf(response_body, response_size,
            "{"
            "\"object\":\"list\","
            "\"data\":[{"
                "\"id\":\"%s\","
                "\"object\":\"model\","
                "\"created\":%ld,"
                "\"owned_by\":\"calm\""
            "}]"
            "}", ctx->display_name, (long)time(NULL));
        return 0;
    }

    if (strcmp(method, "POST") != 0) {
        *out_status_code = 405;
        snprintf(response_body, response_size, "{\"error\":\"method not allowed\"}");
        return 0;
    }

    /* Parse common params */
    char prompt_buf[CALM_MAX_STRING] = {0};
    int is_chat = 0;

    if (strcmp(path, "/v1/chat/completions") == 0) {
        /* Parse messages array → ChatML prompt */
        if (!format_chatml(body, prompt_buf, sizeof(prompt_buf)) || !prompt_buf[0]) {
            *out_status_code = 400;
            snprintf(response_body, response_size,
                     "{\"error\":\"missing or invalid 'messages' field\"}");
            return 0;
        }
        is_chat = 1;
    } else if (strcmp(path, "/v1/completions") == 0) {
        const char* pv = ct_json_get_string(body, "prompt");
        if (!pv) {
            *out_status_code = 400;
            snprintf(response_body, response_size,
                     "{\"error\":\"missing 'prompt' field\"}");
            return 0;
        }
        size_t pl = strlen(pv);
        if (pl >= sizeof(prompt_buf)) pl = sizeof(prompt_buf) - 1;
        memcpy(prompt_buf, pv, pl);
        prompt_buf[pl] = '\0';
    } else {
        *out_status_code = 404;
        snprintf(response_body, response_size, "{\"error\":\"not found\"}");
        return 0;
    }

    int max_tokens = ct_json_get_int(body, "max_tokens", 64);
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 512) max_tokens = 512;

    float temperature = ct_json_get_float(body, "temperature", 0.7f);
    float top_p = ct_json_get_float(body, "top_p", 0.95f);
    float top_k = ct_json_get_float(body, "top_k", 40.0f);
    float repeat_penalty = ct_json_get_float(body, "repeat_penalty", 1.1f);
    (void)ct_json_get_int(body, "echo", 0); /* echo not yet implemented */

    /* JSON mode: check for response_format.type == "json_object" */
    int json_mode = 0;
    const char* rf_val = ct_json_get_value(body, "response_format");
    if (rf_val) {
        char rf_copy[2048];
        size_t rfl = strlen(rf_val);
        if (rfl >= sizeof(rf_copy)) rfl = sizeof(rf_copy) - 1;
        memcpy(rf_copy, rf_val, rfl);
        rf_copy[rfl] = '\0';
        const char* rft = ct_json_get_string(rf_copy, "type");
        if (rft && strcmp(rft, "json_object") == 0) {
            json_mode = 1;
            /* Inject JSON instruction into prompt */
            if (is_chat && strstr(prompt_buf, "<|im_start|>system") == NULL) {
                char tmp[CALM_MAX_STRING];
                snprintf(tmp, sizeof(tmp),
                    "<|im_start|>system\nYou must respond in valid JSON format only, with no additional text before or after the JSON.<|im_end|>\n%s",
                    prompt_buf);
                strncpy(prompt_buf, tmp, sizeof(prompt_buf) - 1);
                prompt_buf[sizeof(prompt_buf) - 1] = '\0';
            } else if (!is_chat) {
                char tmp[CALM_MAX_STRING];
                snprintf(tmp, sizeof(tmp),
                    "Return your response in valid JSON format.\n\n%s",
                    prompt_buf);
                strncpy(prompt_buf, tmp, sizeof(prompt_buf) - 1);
                prompt_buf[sizeof(prompt_buf) - 1] = '\0';
            }
        }
    }

    return gen_completions(ctx, prompt_buf, response_body, response_size,
                           out_status_code, out_content_type,
                           max_tokens, temperature, top_p, top_k,
                           repeat_penalty, is_chat, json_mode);
}

/* Generate a completion (shared by both /v1/completions and /v1/chat/completions) */
static int gen_completions(serve_ctx* ctx, const char* prompt,
                           char* response_body, size_t response_size,
                           int* out_status_code, const char** out_content_type,
                           int max_tokens, float temperature, float top_p,
                           float top_k, float repeat_penalty, int is_chat,
                           int json_mode) {

    CalmGenerateParams params = {
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = repeat_penalty,
        .max_tokens = max_tokens,
        .stream = false,
        .json_mode = (bool)json_mode,
        .tools = (ctx->tools && ctx->tools->count > 0) ? ctx->tools : NULL,
        .max_tool_rounds = 5,
    };

    char output[CALM_MAX_OUTPUT] = {0};
    CalmError err;

    pthread_mutex_lock(&ctx->mutex);
    err = calm_model_generate(ctx->model, prompt, output, sizeof(output), &params);
    pthread_mutex_unlock(&ctx->mutex);

    if (err != CALM_OK) {
        *out_status_code = 500;
        snprintf(response_body, response_size, "{\"error\":\"generation failed\"}");
        return 0;
    }

    int completion_tokens = ctx->model->tokens_generated;
    if (completion_tokens < 0) completion_tokens = 0;
    int prompt_tokens = ctx->model->prompt_tokens;
    if (prompt_tokens < 0) prompt_tokens = 0;

    time_t now = time(NULL);
    char escaped[CALM_MAX_OUTPUT * 2];
    json_escape(output, escaped, sizeof(escaped));

    if (is_chat) {
        /* OpenAI /v1/chat/completions format */
        snprintf(response_body, response_size,
            "{"
            "\"id\":\"chatcmpl-%ld\","
            "\"created\":%ld,"
            "\"model\":\"%s\","
            "\"choices\":[{"
                "\"index\":0,"
                "\"message\":{"
                    "\"role\":\"assistant\","
                    "\"content\":\"%s\""
                "},"
                "\"finish_reason\":\"stop\""
            "}],"
            "\"usage\":{"
                "\"prompt_tokens\":%d,"
                "\"completion_tokens\":%d,"
                "\"total_tokens\":%d"
            "}"
            "}",
            (long)now, (long)now,
            ctx->display_name,
            escaped,
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    } else {
        /* OpenAI /v1/completions format */
        snprintf(response_body, response_size,
            "{"
            "\"id\":\"cmpl-%ld\","
            "\"object\":\"text_completion\","
            "\"created\":%ld,"
            "\"model\":\"%s\","
            "\"choices\":[{"
                "\"text\":\"%s\","
                "\"index\":0,"
                "\"finish_reason\":\"stop\""
            "}],"
            "\"usage\":{"
                "\"prompt_tokens\":%d,"
                "\"completion_tokens\":%d,"
                "\"total_tokens\":%d"
            "}"
            "}",
            (long)now, (long)now,
            ctx->display_name,
            escaped,
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    }

    *out_status_code = 200;
    *out_content_type = "application/json";
    return 0;
}

typedef struct {
    int fd;
    int index;
    char buf[64]; /* partial token buffer for cleanup */
} sse_write_ctx;

static void sse_event_cb(const char* text, void* user_data) {
    sse_write_ctx* sse = (sse_write_ctx*)user_data;
    if (!text || !text[0]) return;
    /* JSON-escape the token text */
    char escaped[2048];
    json_escape(text, escaped, sizeof(escaped));
    char sse_buf[4096];
    int n = snprintf(sse_buf, sizeof(sse_buf),
        "data: {\"choices\":[{\"text\":\"%s\",\"index\":%d}]}\n\n",
        escaped, sse->index);
    if (n > 0) write(sse->fd, sse_buf, (size_t)n);
}

int calm_serve_sse(int fd, const char* body, void* user_data) {
    serve_ctx* ctx = (serve_ctx*)user_data;
    if (!body || !body[0]) {
        static const char err[] = "data: {\"error\":\"no request body\"}\n\n";
        write(fd, err, strlen(err));
        return -1;
    }

    char prompt[4096] = {0};

    /* /v1/completions style: "prompt" field */
    const char* pv = ct_json_get_string(body, "prompt");
    if (pv) {
        size_t pl = strlen(pv);
        if (pl >= sizeof(prompt)) pl = sizeof(prompt) - 1;
        memcpy(prompt, pv, pl);
        prompt[pl] = '\0';
    }

    /* /v1/chat/completions style: "messages" array → ChatML */
    if (!prompt[0] && ct_json_get_value(body, "messages")) {
        format_chatml(body, prompt, sizeof(prompt));
    }

    if (!prompt[0]) {
        static const char err[] = "data: {\"error\":\"missing prompt or messages\"}\n\n";
        write(fd, err, strlen(err));
        return -1;
    }

    int max_tokens = ct_json_get_int(body, "max_tokens", 64);
    if (max_tokens < 1) max_tokens = 1;
    if (max_tokens > 512) max_tokens = 512;
    float temperature = ct_json_get_float(body, "temperature", 0.7f);
    float top_p = ct_json_get_float(body, "top_p", 0.95f);
    float top_k = ct_json_get_float(body, "top_k", 40.0f);
    float repeat_penalty = ct_json_get_float(body, "repeat_penalty", 1.1f);

    /* JSON mode for SSE */
    int sse_json_mode = 0;
    const char* sse_rf_val = ct_json_get_value(body, "response_format");
    if (sse_rf_val) {
        char rf_copy[2048];
        size_t rfl = strlen(sse_rf_val);
        if (rfl >= sizeof(rf_copy)) rfl = sizeof(rf_copy) - 1;
        memcpy(rf_copy, sse_rf_val, rfl);
        rf_copy[rfl] = '\0';
        const char* rft = ct_json_get_string(rf_copy, "type");
        if (rft && strcmp(rft, "json_object") == 0) sse_json_mode = 1;
    }

    sse_write_ctx sse_ctx = { .fd = fd, .index = 0 };

    CalmGenerateParams params = {
        .temperature = temperature,
        .top_p = top_p,
        .top_k = (int)top_k,
        .repeat_penalty = repeat_penalty,
        .max_tokens = max_tokens,
        .stream = true,
        .json_mode = (bool)sse_json_mode,
        .on_token = sse_event_cb,
        .user_data = &sse_ctx,
        .tools = (ctx->tools && ctx->tools->count > 0) ? ctx->tools : NULL,
        .max_tool_rounds = 5,
    };

    char output[CALM_MAX_OUTPUT] = {0};
    CalmError err;

    pthread_mutex_lock(&ctx->mutex);
    err = calm_model_generate(ctx->model, prompt, output, sizeof(output), &params);
    pthread_mutex_unlock(&ctx->mutex);

    /* Send [DONE] marker */
    write(fd, "data: [DONE]\n\n", 14);

    if (err != CALM_OK) return -1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * 10. CLI
 * ═══════════════════════════════════════════════════════════════ */

static void print_usage(const char* prog) {
    printf("Calm v%s — Universal Local LLM Runtime\n\n", calm_version_string());
    printf("Usage:\n");
    printf("  %s scan                    Scan device\n", prog);
    printf("  %s analyze <model.gguf>    Analyze model\n", prog);
    printf("  %s run <model.gguf>        Smart launch\n", prog);
    printf("       [prompt]               Prompt text (default: \"The capital of France is\")\n");
    printf("       --tools <file.json>    Tool definitions (function calling)\n");
    printf("       --temp F               Temperature (default: 0.0)\n");
    printf("       --top-p F              Top-p sampling (default: 0.95)\n");
    printf("       --top-k N              Top-k sampling (default: 40)\n");
    printf("       --repeat-penalty F     Repeat penalty (default: 1.1)\n");
    printf("       --max-tokens N         Max tokens to generate (default: 20, max: 512)\n");
    printf("  %s serve <model.gguf>      Start OpenAI-compatible HTTP server\n", prog);
    printf("       --port <port>          Port (default: 8080)\n");
    printf("       --name <name>          Model display name (default: filename)\n");
    printf("       --tools <file.json>    Tool definitions (function calling)\n");
    printf("       (generation params set via JSON request body: temperature, top_p, top_k, repeat_penalty, max_tokens)\n");
    printf("  %s estimate <model.gguf>   Performance prediction\n", prog);
    printf("  %s convert <model.gguf>    Convert to 1-bit (BQ1_0/TQ1_0)\n", prog);
    printf("       --format bq1_0|tq1_0  Target format (default: bq1_0)\n");
    printf("       --output <file.gguf>  Output path\n");
    printf("  %s model info <model.gguf> Model analysis\n", prog);
    printf("  %s tokenize <model.gguf> <text>  Tokenize & decode text\n", prog);
    printf("  %s download <url>         Download model from URL or HF repo ID\n", prog);
    printf("  %s --help                  This help\n", prog);
    printf("  %s --version               Version info\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0) {
        printf("Calm v%s\n", calm_version_string());
        return 0;
    }

    // Global device probe (lazy, reused)
    static CalmDevice cached_device;
    static bool device_cached = false;

    if (!device_cached) {
        calm_device_probe(&cached_device);
        device_cached = true;
    }

    // ── model ──
    if (strcmp(argv[1], "model") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s model info <model.gguf>\n", argv[0]);
            return 1;
        }
        const char* sub = argv[2];
        const char* mpath = argv[3];
        if (access(mpath, F_OK) != 0) {
            fprintf(stderr, "Error: file not found: %s\n", mpath);
            return 1;
        }
        if (strcmp(sub, "info") == 0) {
            CalmModelInfo minfo;
            CalmError merr = calm_model_read_info(mpath, &minfo);
            if (merr != CALM_OK) {
                fprintf(stderr, "Error reading model: %s\n", calm_err_string(merr));
                return 1;
            }
            CalmPlan mplan;
            calm_plan_create(&cached_device, &minfo, &mplan);
            print_model_info(&minfo, &mplan);
            calm_model_info_free(&minfo);
            return 0;
        }
        fprintf(stderr, "Unknown model subcommand: %s\n", sub);
        return 1;
    }

    // ── tokenize ──
    if (strcmp(argv[1], "tokenize") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s tokenize <model.gguf> <text>\n", argv[0]);
            fprintf(stderr, "       %s tokenize <model.gguf> -   (read text from stdin)\n", argv[0]);
            return 1;
        }
        const char* tpath = argv[2];
        if (access(tpath, F_OK) != 0) {
            fprintf(stderr, "Error: file not found: %s\n", tpath);
            return 1;
        }
        // Get text from arg or stdin
        char text[65536] = {0};
        if (strcmp(argv[3], "-") == 0) {
            size_t n = fread(text, 1, sizeof(text) - 1, stdin);
            text[n] = '\0';
            // Trim trailing newline
            while (n > 0 && (text[n-1] == '\n' || text[n-1] == '\r')) text[--n] = '\0';
        } else {
            size_t remaining = sizeof(text) - 1;
            text[0] = '\0';
            for (int i = 3; i < argc && remaining > 0; i++) {
                size_t l = strlen(argv[i]);
                if (l + 1 > remaining) l = remaining - 1;
                if (i > 3) { strncat(text, " ", remaining); remaining--; }
                strncat(text, argv[i], remaining);
                if (l < remaining) remaining -= l;
                else remaining = 0;
            }
        }
        // Open GGUF and tokenizer
        ct_gguf_context* tgguf = ct_gguf_open(tpath);
        if (!tgguf) {
            fprintf(stderr, "Error: failed to open GGUF file\n");
            return 1;
        }
        ct_tokenizer* tok = ct_tokenizer_load(tgguf);
        if (!tok) {
            fprintf(stderr, "Error: model has no tokenizer data\n");
            ct_gguf_close(tgguf);
            return 1;
        }
        printf("Tokenizer: %s\n", tok->type == CT_TOKENIZER_BPE ? "BPE" : "Unknown");
        printf("Vocab: %d tokens, %d merges\n", tok->vocab_size, tok->merge_count);
        printf("BOS=%d EOS=%d add_bos=%d\n", tok->bos_id, tok->eos_id, tok->add_bos);
        printf("\nInput text: \"%s\"\n\n", text);
        // Encode
        int tokens[4096];
        int n_tok = ct_tokenizer_encode(tok, text, tokens, 4096);
        if (n_tok < 1) {
            printf("(no tokens produced)\n");
        } else {
            printf("Tokens (%d): ", n_tok);
            for (int i = 0; i < n_tok; i++) {
                printf(" %d", tokens[i]);
            }
            printf("\n\nToken strings:\n");
            for (int i = 0; i < n_tok; i++) {
                int id = tokens[i];
                if (id >= 0 && id < tok->vocab_size) {
                    char esc[256] = {0};
                    size_t ei = 0;
                    for (size_t j = 0; j < (size_t)tok->token_lens[id] && ei < sizeof(esc) - 4; j++) {
                        unsigned char c = tok->tokens[id][j];
                        if (c == '\n') { esc[ei++] = '\\'; esc[ei++] = 'n'; }
                        else if (c == '\t') { esc[ei++] = '\\'; esc[ei++] = 't'; }
                        else if (c == ' ') { esc[ei++] = '\xe2'; esc[ei++] = '\x96'; esc[ei++] = '\x81'; } /* ▁ */
                        else if (c < 32 || c > 126) { snprintf(esc + ei, sizeof(esc) - ei, "\\x%02x", c); ei = strlen(esc); }
                        else { esc[ei++] = c; }
                    }
                    esc[ei] = '\0';
                    printf("  %4d: \"%s\"\n", id, esc);
                } else {
                    printf("  %4d: (out of range)\n", id);
                }
            }
            // Decode
            char decoded[4096];
            ct_tokenizer_decode(tok, tokens, n_tok, decoded, sizeof(decoded));
            printf("\nDecoded: \"%s\"\n", decoded);
            if (strcmp(text, decoded) == 0) {
                printf("✓ Roundtrip: EXACT MATCH\n");
            } else {
                printf("⚠ Roundtrip: MISMATCH (\"%s\" → \"%s\")\n", text, decoded);
            }
        }
        ct_tokenizer_free(tok);
        ct_gguf_close(tgguf);
        return 0;
    }

    // ── scan ──
    if (strcmp(argv[1], "scan") == 0) {
        print_device_info(&cached_device);

        // Recommendations
        printf("%sWhat can run on this device%s\n", C(ANSI_BOLD), C(ANSI_RESET));
        printf("───────────────────────────────────────────────────────\n");

        double ram = cached_device.ram_available / (double)(1 << 30);

        if (ram >= 16)
            printf("  🚀 High-end:  70B Q4, 27B FP16, DeepSeek-V3\n");
        else if (ram >= 8)
            printf("  💪 Great:      27B Q4, 13B Q8, Mixtral 8x7B\n");
        else if (ram >= 6)
            printf("  👍 Good:       13B Q4, 9B Q4_K_M, 7B Q8\n");
        else if (ram >= 4)
            printf("  ✅ Decent:     7B Q4_K_M, 3B Q8, Bonsai 27B 1-bit\n");
        else if (ram >= 3)
            printf("  ⚠️  Tight:     Bonsai 27B 1-bit, 3B Q4_K_M, 1.5B Q8\n");
        else if (ram >= 2)
            printf("  🟡 Limited:   3B Q4, 1.5B Q4_K_M\n");
        else
            printf("  🔴 Very limited: 0.5B Q4 only\n");

        if (ram >= 2)
            printf("  🌊 MoE stream: GLM-5.2 744B, Mixtral 8x7B (Colibri-style)\n");
        if (ram >= 3)
            printf("  🌀 1-bit:      27B 1-bit (3.9 GB)\n");

        printf("\n");
        return 0;
    }

    // ── download ──
    if (strcmp(argv[1], "download") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s download <url> [output_filename]\n", argv[0]);
            fprintf(stderr, "Downloads a model from Hugging Face or any HTTP(S) URL.\n");
            fprintf(stderr, "  url    — Hugging Face repo ID or full URL\n");
            fprintf(stderr, "  output — optional output filename (default: derived from URL)\n");
            return 1;
        }
        const char* src = argv[2];
        char url[4096];
        if (strstr(src, "://")) {
            strncpy(url, src, sizeof(url) - 1);
        } else {
            snprintf(url, sizeof(url),
                "https://huggingface.co/%s/resolve/main/model.gguf", src);
        }
        char output[4096] = {0};
        if (argc >= 4) {
            strncpy(output, argv[3], sizeof(output) - 1);
        } else {
            const char* last_slash = strrchr(url, '/');
            if (last_slash)
                snprintf(output, sizeof(output), "%s", last_slash + 1);
            else
                snprintf(output, sizeof(output), "model.gguf");
        }
        printf("Downloading: %s\n", url);
        printf("Output:      %s\n\n", output);
        fflush(stdout);
        char cmd[8192];
        snprintf(cmd, sizeof(cmd), "curl -L --progress-bar -o '%s' '%s'", output, url);
        int rc = system(cmd);
        if (rc == 0) {
            printf("\n✓ Downloaded to %s\n", output);
        } else {
            fprintf(stderr, "\n✗ Download failed (curl exit code %d)\n", rc);
            fprintf(stderr, "Tip: Install curl or use: wget -O '%s' '%s'\n", output, url);
            return 1;
        }
        return 0;
    }

    // Все остальные команды требуют путь к модели
    if (argc < 3) {
        fprintf(stderr, "Error: model path required\n");
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[2];

    // Проверяем, существует ли файл
    if (access(model_path, F_OK) != 0) {
        fprintf(stderr, "Error: file not found: %s\n", model_path);
        return 1;
    }

    // Читаем информацию о модели
    CalmModelInfo info;
    CalmError err = calm_model_read_info(model_path, &info);
    if (err != CALM_OK) {
        fprintf(stderr, "Error reading model: %s\n", calm_err_string(err));
        return 1;
    }

    // Создаём план
    CalmPlan plan;
    calm_plan_create(&cached_device, &info, &plan);

    // ── analyze ──
    if (strcmp(argv[1], "analyze") == 0) {
        print_model_info(&info, &plan);
        return 0;
    }

    // ── estimate ──
    if (strcmp(argv[1], "estimate") == 0) {
        print_model_info(&info, &plan);
        print_performance_estimate(&cached_device, &info, &plan);
        return 0;
    }

    // ── convert ──
    if (strcmp(argv[1], "convert") == 0) {
        const char* format = "bq1_0";
        const char* output = NULL;

        // Parse extra args
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
                format = argv[++i];
            else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
                output = argv[++i];
        }

        // Auto-generate output filename if not specified
        char auto_output[4096];
        if (!output) {
            snprintf(auto_output, sizeof(auto_output), "%s", model_path);
            char* dot = strrchr(auto_output, '.');
            if (dot && (strcmp(dot, ".gguf") == 0 || strcmp(dot, ".GGUF") == 0))
                *dot = '\0';
            strcat(auto_output, strcmp(format, "tq1_0") == 0 ? "-tq1_0.gguf" : "-bq1_0.gguf");
            output = auto_output;
        }

        // Find calm_convert binary (same directory as calm)
        char converter_path[4096];
        const char* self_dir = strrchr(argv[0], '/');
        if (self_dir) {
            size_t dir_len = (size_t)(self_dir - argv[0]);
            snprintf(converter_path, sizeof(converter_path), "%.*s/calm_convert",
                     (int)dir_len, argv[0]);
        } else {
            snprintf(converter_path, sizeof(converter_path), "calm_convert");
        }

        printf("Converting %s → %s (%s)...\n\n", model_path, output, format);
        fflush(stdout);

        char cmd[8192];
        snprintf(cmd, sizeof(cmd), "%s --input '%s' --format %s --output '%s'",
                 converter_path, model_path, format, output);

        int rc = system(cmd);
        if (rc == -1) {
            fprintf(stderr, "Error: calm_convert not found at %s\n"
                    "Build it: make calm_convert\n", converter_path);
            return 1;
        }
        return rc;
    }

    // ── run ──
    if (strcmp(argv[1], "run") == 0) {
        print_model_info(&info, &plan);

        printf("%sLaunching...%s\n\n", C(ANSI_BOLD), C(ANSI_RESET));

        // Создаём runtime
        CalmRuntime* rt = calm_init(&cached_device);
        if (!rt) {
            fprintf(stderr, "Failed to initialize runtime\n");
            return 1;
        }

        CalmModel* model = calm_model_load(rt, model_path, &plan);
        if (!model) {
            calm_destroy(rt);
            return 1;
        }

        // Parse extra args
        const char* prompt = "The capital of France is";
        const char* tools_path = NULL;
        float temp = 0.0f;
        float top_p = 0.95f;
        float top_k = 40.0f;
        float repeat_penalty = 1.1f;
        int max_tokens = 20;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--tools") == 0 && i + 1 < argc) {
                tools_path = argv[++i];
            } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
                temp = (float)atof(argv[++i]);
            } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
                temp = (float)atof(argv[++i]);
            } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
                top_p = (float)atof(argv[++i]);
            } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
                top_k = (float)atof(argv[++i]);
            } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
                repeat_penalty = (float)atof(argv[++i]);
            } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
                if (max_tokens < 1) max_tokens = 1;
                if (max_tokens > 512) max_tokens = 512;
            } else if (prompt[0] == 'T' || i == 3) {
                /* First non-flag arg is the prompt */
                if (argv[i][0] != '-') {
                    prompt = argv[i];
                }
            }
        }

        // Load tool definitions if --tools specified
        CalmToolDefinitions tool_defs = {0};
        char tools_file_buf[CT_TOOL_PARAMS_MAX * CT_TOOLS_MAX];
        if (tools_path) {
            FILE* tf = fopen(tools_path, "r");
            if (tf) {
                size_t n = fread(tools_file_buf, 1, sizeof(tools_file_buf) - 1, tf);
                fclose(tf);
                tools_file_buf[n] = '\0';
                // Remove BOM if present
                char* json_start = tools_file_buf;
                if ((unsigned char)json_start[0] == 0xEF &&
                    (unsigned char)json_start[1] == 0xBB &&
                    (unsigned char)json_start[2] == 0xBF)
                    json_start += 3;
                ct_tools_parse_definitions(json_start, &tool_defs);
                fprintf(stderr, "[TOOLS] loaded %d tool(s) from %s\n",
                        tool_defs.count, tools_path);
            } else {
                fprintf(stderr, "[TOOLS] cannot open %s\n", tools_path);
            }
        }

        // Generate
        char output[CALM_MAX_OUTPUT] = {0};
        CalmGenerateParams params = {
            .temperature = temp,
            .top_p = top_p,
            .top_k = top_k,
            .repeat_penalty = repeat_penalty,
            .max_tokens = max_tokens,
            .stream = false,
            .tools = (tool_defs.count > 0) ? &tool_defs : NULL,
            .max_tool_rounds = 5,
        };

        err = calm_model_generate(model, prompt, output, sizeof(output), &params);
        if (err == CALM_OK) {
            printf("%s\n", output);
        }

        calm_model_free(model);
        calm_destroy(rt);

        return err == CALM_OK ? 0 : 1;
    }

    // ── serve ──
    if (strcmp(argv[1], "serve") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s serve <model.gguf> [--port PORT] [--tools tools.json]\n", argv[0]);
            return 1;
        }

        print_model_info(&info, &plan);
        printf("%sStarting server...%s\n\n", C(ANSI_BOLD), C(ANSI_RESET));

        // Init runtime
        CalmRuntime* rt = calm_init(&cached_device);
        if (!rt) {
            fprintf(stderr, "Failed to initialize runtime\n");
            return 1;
        }

        CalmModel* model = calm_model_load(rt, model_path, &plan);
        if (!model) {
            calm_destroy(rt);
            return 1;
        }

        // Parse extra args
        int port = 8080;
        const char* tools_path = NULL;
        const char* model_name_arg = NULL;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
                port = atoi(argv[++i]);
                if (port <= 0 || port > 65535) port = 8080;
            } else if (strcmp(argv[i], "--tools") == 0 && i + 1 < argc) {
                tools_path = argv[++i];
            } else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
                model_name_arg = argv[++i];
            }
        }

        // Load tool definitions if --tools specified
        CalmToolDefinitions tool_defs = {0};
        char tools_file_buf[CT_TOOL_PARAMS_MAX * CT_TOOLS_MAX];
        if (tools_path) {
            FILE* tf = fopen(tools_path, "r");
            if (tf) {
                size_t n = fread(tools_file_buf, 1, sizeof(tools_file_buf) - 1, tf);
                fclose(tf);
                tools_file_buf[n] = '\0';
                char* json_start = tools_file_buf;
                if ((unsigned char)json_start[0] == 0xEF &&
                    (unsigned char)json_start[1] == 0xBB &&
                    (unsigned char)json_start[2] == 0xBF)
                    json_start += 3;
                ct_tools_parse_definitions(json_start, &tool_defs);
                fprintf(stderr, "[TOOLS] loaded %d tool(s) from %s\n",
                        tool_defs.count, tools_path);
            } else {
                fprintf(stderr, "[TOOLS] cannot open %s\n", tools_path);
            }
        }

        // Build server context
        serve_ctx ctx;
        ctx.model = model;
        ctx.tools = (tool_defs.count > 0) ? &tool_defs : NULL;
        pthread_mutex_init(&ctx.mutex, NULL);
        // Set display name: use --name if given, else basename of model path
        if (model_name_arg) {
            snprintf(ctx.display_name, sizeof(ctx.display_name), "%s", model_name_arg);
        } else {
            const char* base = strrchr(model_path, '/');
            base = base ? base + 1 : model_path;
            snprintf(ctx.display_name, sizeof(ctx.display_name), "%s", base);
            char* dot = strrchr(ctx.display_name, '.');
            if (dot) *dot = '\0';
        }

        // Start server (blocks until SIGINT)
        int ret = ct_server_start(port, handle_v1_completions, &ctx);

        pthread_mutex_destroy(&ctx.mutex);
        calm_model_free(model);
        calm_destroy(rt);

        return ret == 0 ? 0 : 1;
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    print_usage(argv[0]);
    return 1;
}

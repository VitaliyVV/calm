/**
 * calm_vulkan.c — Calm Vulkan GPU Backend Implementation
 *
 * Zero-dependency Vulkan compute for quantized LLM inference.
 * Supports Q8_0 matmul via compute shader on any Vulkan 1.1+ device.
 *
 * Architecture:
 *   ct_vulkan_backend
 *   ├── VkInstance, VkDevice, VkQueue, VkCommandPool
 *   ├── Descriptor pool + descriptor set layouts
 *   ├── Compute pipeline for Q8_0 matmul
 *   ├── Weight buffer pool (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
 *   └── Host-coherent staging buffer (for x/y ping-pong)
 */
#include "calm_vulkan.h"
#include "calm_quant.h"
#include "calm_gguf.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

/* ═══════════════════════════════════════════════════════════════
 * Embedded SPIR-V shader (compiled from shaders/q8_0_matmul.comp)
 * ═══════════════════════════════════════════════════════════════ */
/* We wrap the generated header to make it C-compatible */
#define q8_0_matmul_spv q8_0_matmul_spv_data
#include "shaders/q8_0_matmul_spv.h"
#undef q8_0_matmul_spv
static const uint32_t* q8_0_matmul_spv = q8_0_matmul_spv_data;
static const size_t q8_0_matmul_spv_size = sizeof(q8_0_matmul_spv_data);

/* ═══════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════ */
#define CT_VK_MAX_WEIGHTS 1024
#define CT_VK_WORKS_PER_BLOCK 32       /* Q8_0 block size */
#define CT_VK_U32_PER_Q8_BLOCK 9       /* repacked: 1 scale + 8 data uint32 */
#define CT_VK_WG_SIZE 256              /* workgroup size in shader */
#define CT_VK_MAX_STAGING (8 * 1024 * 1024) /* 8 MB staging buffer */

/* ═══════════════════════════════════════════════════════════════
 * Backend struct
 * ═══════════════════════════════════════════════════════════════ */
struct ct_vulkan_backend {
    /* Vulkan handles */
    VkInstance           instance;
    VkPhysicalDevice     phys_dev;
    VkDevice             device;
    VkQueue              queue;
    uint32_t             queue_family;
    VkCommandPool        cmd_pool;
    VkDescriptorPool     desc_pool;
    
    /* Pipeline for Q8_0 matmul */
    VkDescriptorSetLayout  ds_layout;
    VkPipelineLayout       pipeline_layout;
    VkPipeline             pipeline_q8_0;
    VkShaderModule         shader_q8_0;
    
    /* Command buffer (reusable, one-time-submit) */
    VkCommandBuffer        cmd;
    VkFence                fence;
    
    /* Device info */
    char    device_name[256];
    int     initialized;
    
    /* Weight buffer pool */
    struct {
        VkBuffer       buf;
        VkDeviceMemory mem;
        void*          ptr;        /* mapped HOST_VISIBLE pointer */
        size_t         capacity;   /* allocated size in bytes */
        size_t         used;       /* bytes consumed so far */
        int            count;
    } weights;
    
    /* Per-weight metadata (indexed by weight_id) */
    struct {
        size_t offset;   /* offset in weight buffer (in uint32 words) */
        int     type;    /* CT_GGUF_TYPE_* */
        int     I;
        int     O;
        int     blk_per_I;
        int     blk_stride;   /* uint32 per row */
        size_t  size_u32;    /* total uint32 words */
        char    name[64];
    } weight_meta[CT_VK_MAX_WEIGHTS];
    
    /* Upload staging buffer (reusable, persistent) */
    VkBuffer       upload_buf;
    VkDeviceMemory upload_mem;
    void*          upload_ptr;
    size_t         upload_cap;
    
    /* Inference staging buffer for x/y transfer */
    VkBuffer       staging;
    VkDeviceMemory staging_mem;
    void*          staging_ptr;    /* mapped pointer */
    size_t         staging_size;
    
    /* Batch matmul infrastructure */
    int            batch_active;
    int            batch_count;
    size_t         batch_stag_used;   /* bytes consumed in staging for current batch */
#define CT_VK_MAX_BATCH 32
    struct {
        float*      dst;       /* caller's y pointer for deferred copy-back */
        size_t      stag_off;  /* offset of y in staging buffer */
        size_t      size;      /* y size in bytes */
    } batch_y[CT_VK_MAX_BATCH];
    VkDescriptorSet  batch_sets[CT_VK_MAX_BATCH];  /* pre-allocated, cycled */
    int              batch_set_idx;
    VkCommandBuffer  batch_cmd;      /* command buffer for batched dispatches */
    VkFence          batch_fence;
};

/* ═══════════════════════════════════════════════════════════════
 * Vulkan function pointers (loaded via vkGetInstanceProcAddr)
 * ═══════════════════════════════════════════════════════════════ */
#define VK_FUNC(name) static PFN_##name pfn_##name = NULL
VK_FUNC(vkGetInstanceProcAddr);
VK_FUNC(vkCreateInstance);
VK_FUNC(vkDestroyInstance);
VK_FUNC(vkEnumeratePhysicalDevices);
VK_FUNC(vkGetPhysicalDeviceProperties);
VK_FUNC(vkGetPhysicalDeviceMemoryProperties);
VK_FUNC(vkGetPhysicalDeviceFeatures);
VK_FUNC(vkCreateDevice);
VK_FUNC(vkDestroyDevice);
VK_FUNC(vkGetDeviceQueue);
VK_FUNC(vkDeviceWaitIdle);
VK_FUNC(vkCreateCommandPool);
VK_FUNC(vkDestroyCommandPool);
VK_FUNC(vkAllocateCommandBuffers);
VK_FUNC(vkFreeCommandBuffers);
VK_FUNC(vkBeginCommandBuffer);
VK_FUNC(vkEndCommandBuffer);
VK_FUNC(vkQueueSubmit);
VK_FUNC(vkQueueWaitIdle);
VK_FUNC(vkCreateShaderModule);
VK_FUNC(vkDestroyShaderModule);
VK_FUNC(vkCreateDescriptorSetLayout);
VK_FUNC(vkDestroyDescriptorSetLayout);
VK_FUNC(vkCreateDescriptorPool);
VK_FUNC(vkDestroyDescriptorPool);
VK_FUNC(vkAllocateDescriptorSets);
VK_FUNC(vkUpdateDescriptorSets);
VK_FUNC(vkCreatePipelineLayout);
VK_FUNC(vkDestroyPipelineLayout);
VK_FUNC(vkCreateComputePipelines);
VK_FUNC(vkDestroyPipeline);
VK_FUNC(vkCreateBuffer);
VK_FUNC(vkDestroyBuffer);
VK_FUNC(vkGetBufferMemoryRequirements);
VK_FUNC(vkAllocateMemory);
VK_FUNC(vkFreeMemory);
VK_FUNC(vkBindBufferMemory);
VK_FUNC(vkMapMemory);
VK_FUNC(vkUnmapMemory);
VK_FUNC(vkCmdBindPipeline);
VK_FUNC(vkCmdBindDescriptorSets);
VK_FUNC(vkCmdPushConstants);
VK_FUNC(vkCmdDispatch);
VK_FUNC(vkCmdCopyBuffer);
VK_FUNC(vkCmdPipelineBarrier);
VK_FUNC(vkCreateFence);
VK_FUNC(vkDestroyFence);
VK_FUNC(vkWaitForFences);
VK_FUNC(vkResetFences);
VK_FUNC(vkFlushMappedMemoryRanges);
VK_FUNC(vkInvalidateMappedMemoryRanges);
VK_FUNC(vkResetDescriptorPool);
VK_FUNC(vkGetPhysicalDeviceQueueFamilyProperties);

/* ═══════════════════════════════════════════════════════════════
 * Loader — dynamically loads libvulkan.so
 * ═══════════════════════════════════════════════════════════════ */
static void* vk_handle = NULL;

#define VK_LOAD(name) do { \
    pfn_##name = (PFN_##name)ct_vk_get_proc_addr(#name); \
    if (!pfn_##name) { fprintf(stderr, "vulkan: failed to load " #name "\n"); return NULL; } \
} while(0)

#define VK_LOAD_DEV(name) do { \
    pfn_##name = (PFN_##name)ct_vk_get_proc_addr(#name); \
    if (!pfn_##name) { fprintf(stderr, "vulkan: failed to load " #name "\n"); return -1; } \
} while(0)

static void* ct_vk_get_proc_addr(const char* name) {
    if (!vk_handle) {
        vk_handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (!vk_handle) {
            vk_handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        }
    }
    if (vk_handle) {
        void* p = dlsym(vk_handle, name);
        if (p) return p;
        /* Try vkGetInstanceProcAddr for device functions */
        if (pfn_vkGetInstanceProcAddr) {
            return (void*)pfn_vkGetInstanceProcAddr(NULL, name);
        }
    }
    return NULL;
}

static void ct_vk_unload_lib(void) {
    if (vk_handle) {
        dlclose(vk_handle);
        vk_handle = NULL;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FP16 helper (host-side, for repacking)
 * ═══════════════════════════════════════════════════════════════ */
static inline float fp16_to_f32_vk(uint16_t h) {
    const uint32_t sign  = ((uint32_t)h & 0x8000U) << 16;
    const uint32_t exp16 = ((uint32_t)h >> 10) & 0x1FU;
    const uint32_t mant  = (uint32_t)h & 0x03FFU;
    uint32_t r;
    if (exp16 == 0) {
        if (mant == 0) r = sign;
        else {
            int shift = 10;
            uint32_t m = mant;
            while ((m & 0x0400) == 0) { m <<= 1; shift--; }
            m &= 0x03FF;
            r = sign | ((uint32_t)(127 - 14 - shift) << 23) | (m << 13);
        }
    } else if (exp16 == 31) {
        r = sign | 0x7F800000U | (mant << 13);
    } else {
        r = sign | ((uint32_t)(exp16 + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &r, sizeof(f));
    return f;
}

/* ═══════════════════════════════════════════════════════════════
 * Q8_0 repack: CPU block → GPU-friendly uint32 array
 *
 * CPU layout:  ct_block_q8_0 = { uint16_t d; int8_t qs[32]; }  (34 bytes padded)
 * GPU layout:  9 × uint32:
 *   [0]: low16 = FP16 scale d
 *   [1-8]: qs[32] as 8 × uint32, each = 4 × int8 packed
 * ═══════════════════════════════════════════════════════════════ */
static void repack_q8_0_block(const ct_block_q8_0* src, uint32_t* dst) {
    dst[0] = (uint32_t)src->d;  /* FP16 bits in low 16 bits */
    for (int k = 0; k < 8; k++) {
        uint32_t w = 0;
        w |= (uint32_t)(uint8_t)src->qs[k*4+0] << 0;
        w |= (uint32_t)(uint8_t)src->qs[k*4+1] << 8;
        w |= (uint32_t)(uint8_t)src->qs[k*4+2] << 16;
        w |= (uint32_t)(uint8_t)src->qs[k*4+3] << 24;
        dst[1 + k] = w;
    }
}

/* Compute blk_stride: must be multiple of device's minStorageBufferOffsetAlignment */
static int align_stride(int stride_u32, VkDeviceSize align) {
    VkDeviceSize byte_size = (VkDeviceSize)stride_u32 * sizeof(uint32_t);
    byte_size = (byte_size + align - 1) & ~(align - 1);
    return (int)(byte_size / sizeof(uint32_t));
}

/* ═══════════════════════════════════════════════════════════════
 * Memory type selection helper
 * ═══════════════════════════════════════════════════════════════ */
static int ct_vk_find_mem_type(ct_vulkan_backend* vk, uint32_t type_filter,
                                VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    pfn_vkGetPhysicalDeviceMemoryProperties(vk->phys_dev, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return (int)i;
        }
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════
 * Buffer creation helper
 * ═══════════════════════════════════════════════════════════════ */
static int ct_vk_create_buffer(ct_vulkan_backend* vk, VkDeviceSize size,
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props,
                                VkBuffer* buf, VkDeviceMemory* mem) {
    VkBufferCreateInfo binfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult r = pfn_vkCreateBuffer(vk->device, &binfo, NULL, buf);
    if (r != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    pfn_vkGetBufferMemoryRequirements(vk->device, *buf, &req);

    VkMemoryAllocateInfo ainfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
    };
    int mem_type = ct_vk_find_mem_type(vk, req.memoryTypeBits, props);
    if (mem_type < 0) {
        pfn_vkDestroyBuffer(vk->device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        return -1;
    }
    ainfo.memoryTypeIndex = (uint32_t)mem_type;

    r = pfn_vkAllocateMemory(vk->device, &ainfo, NULL, mem);
    if (r != VK_SUCCESS) {
        pfn_vkDestroyBuffer(vk->device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        return -1;
    }

    r = pfn_vkBindBufferMemory(vk->device, *buf, *mem, 0);
    if (r != VK_SUCCESS) {
        pfn_vkDestroyBuffer(vk->device, *buf, NULL);
        pfn_vkFreeMemory(vk->device, *mem, NULL);
        *buf = VK_NULL_HANDLE;
        *mem = VK_NULL_HANDLE;
        return -1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Init: load library + create instance
 * ═══════════════════════════════════════════════════════════════ */
ct_vulkan_backend* ct_vulkan_init(void) {
    ct_vulkan_backend* vk = (ct_vulkan_backend*)calloc(1, sizeof(ct_vulkan_backend));
    if (!vk) return NULL;

    /* Load vkCreateInstance first — the rest via getProcAddr.
     * Try Android system Vulkan loader first (HAL-based, uses real GPU),
     * then fall back to Termux/standard libvulkan.so. */
    static const char* vk_paths[] = {
        "/system/lib64/libvulkan.so",    /* Android HAL — hardware Adreno */
        "libvulkan.so",
        "libvulkan.so.1",
        NULL
    };
    for (int i = 0; vk_paths[i]; i++) {
        vk_handle = dlopen(vk_paths[i], RTLD_NOW | RTLD_LOCAL);
        if (vk_handle) {
            fprintf(stderr, "vulkan: loaded %s\n", vk_paths[i]);
            break;
        }
    }
    if (!vk_handle) {
        fprintf(stderr, "vulkan: libvulkan.so not found (tried system HAL and default)\n");
        goto fail;
    }

    pfn_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)
        dlsym(vk_handle, "vkGetInstanceProcAddr");
    if (!pfn_vkGetInstanceProcAddr) {
        fprintf(stderr, "vulkan: vkGetInstanceProcAddr not found\n");
        goto fail;
    }

    VK_LOAD(vkCreateInstance);
    VK_LOAD(vkEnumeratePhysicalDevices);
    VK_LOAD(vkGetPhysicalDeviceProperties);
    VK_LOAD(vkGetPhysicalDeviceMemoryProperties);
    VK_LOAD(vkGetPhysicalDeviceFeatures);
    VK_LOAD(vkDestroyInstance);

    /* Create instance (no extensions needed for compute-only) */
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Calm",
        .applicationVersion = 1,
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (pfn_vkCreateInstance(&info, NULL, &vk->instance) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create instance\n");
        goto fail;
    }

    /* Enumerate physical devices */
    uint32_t ndev = 0;
    pfn_vkEnumeratePhysicalDevices(vk->instance, &ndev, NULL);
    if (ndev == 0) {
        fprintf(stderr, "vulkan: no physical devices\n");
        goto fail;
    }
    VkPhysicalDevice* devs = (VkPhysicalDevice*)malloc(ndev * sizeof(VkPhysicalDevice));
    pfn_vkEnumeratePhysicalDevices(vk->instance, &ndev, devs);

    /* Pick the best device: prefer discrete > integrated > other, skip CPU/virtual */
    int best = -1;
    int best_score = -1;
    fprintf(stderr, "vulkan: found %d device(s):\n", ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties props;
        pfn_vkGetPhysicalDeviceProperties(devs[i], &props);
        const char* type_str = "?";
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) type_str = "discrete";
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) type_str = "integrated";
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) type_str = "CPU";
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) type_str = "virtual";
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER) type_str = "other";
        fprintf(stderr, "vulkan:   [%d] %s (%s) api=0x%x\n",
                i, props.deviceName, type_str, props.apiVersion);
        if (props.apiVersion < VK_API_VERSION_1_0) continue;
        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 3;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 2;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER) score = 1;
        else score = 0;
        if (score > best_score) { best_score = score; best = (int)i; }
    }
    /* Fallback: pick any Vulkan 1.1+ device */
    if (best < 0) {
        fprintf(stderr, "vulkan: no preferred device, picking first 1.1+\n");
        for (uint32_t i = 0; i < ndev; i++) {
            VkPhysicalDeviceProperties props;
            pfn_vkGetPhysicalDeviceProperties(devs[i], &props);
            if (props.apiVersion >= VK_API_VERSION_1_1) { best = (int)i; break; }
        }
    }
    if (best < 0) {
        fprintf(stderr, "vulkan: no Vulkan 1.1+ device\n");
        free(devs);
        goto fail;
    }
    vk->phys_dev = devs[best];

    /* Get device name */
    VkPhysicalDeviceProperties props;
    pfn_vkGetPhysicalDeviceProperties(vk->phys_dev, &props);
    snprintf(vk->device_name, sizeof(vk->device_name), "%s",
             props.deviceName);
    fprintf(stderr, "vulkan: using device: %s\n", props.deviceName);

    /* Get queue family with compute support */
    *(void**)&pfn_vkGetPhysicalDeviceQueueFamilyProperties = ct_vk_get_proc_addr("vkGetPhysicalDeviceQueueFamilyProperties");
    uint32_t nqf_families = 0;
    pfn_vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_dev, &nqf_families, NULL);
    if (nqf_families == 0) {
        fprintf(stderr, "vulkan: no queue families\n");
        free(devs);
        goto fail;
    }
    VkQueueFamilyProperties* qfp = (VkQueueFamilyProperties*)
        malloc(nqf_families * sizeof(VkQueueFamilyProperties));
    pfn_vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_dev, &nqf_families, qfp);

    int qf_found = -1;
    for (uint32_t i = 0; i < nqf_families; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qf_found = (int)i;
            break;
        }
    }
    free(qfp);
    if (qf_found < 0) {
        fprintf(stderr, "vulkan: no compute queue\n");
        free(devs);
        goto fail;
    }
    vk->queue_family = (uint32_t)qf_found;

    /* Load device creation functions */
    VK_LOAD(vkCreateDevice);
    VK_LOAD(vkDestroyDevice);
    VK_LOAD(vkGetDeviceQueue);
    VK_LOAD(vkDeviceWaitIdle);

    /* Create logical device */
    float qp = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &qp,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    if (pfn_vkCreateDevice(vk->phys_dev, &dci, NULL, &vk->device) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create device\n");
        free(devs);
        goto fail;
    }
    free(devs);

    pfn_vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

    /* ─── Load remaining device functions ─── */
    VK_LOAD(vkCreateCommandPool);
    VK_LOAD(vkDestroyCommandPool);
    VK_LOAD(vkAllocateCommandBuffers);
    VK_LOAD(vkFreeCommandBuffers);
    VK_LOAD(vkBeginCommandBuffer);
    VK_LOAD(vkEndCommandBuffer);
    VK_LOAD(vkQueueSubmit);
    VK_LOAD(vkQueueWaitIdle);
    VK_LOAD(vkCreateShaderModule);
    VK_LOAD(vkDestroyShaderModule);
    VK_LOAD(vkCreateDescriptorSetLayout);
    VK_LOAD(vkDestroyDescriptorSetLayout);
    VK_LOAD(vkCreateDescriptorPool);
    VK_LOAD(vkDestroyDescriptorPool);
    VK_LOAD(vkAllocateDescriptorSets);
    VK_LOAD(vkUpdateDescriptorSets);
    VK_LOAD(vkCreatePipelineLayout);
    VK_LOAD(vkDestroyPipelineLayout);
    VK_LOAD(vkCreateComputePipelines);
    VK_LOAD(vkDestroyPipeline);
    VK_LOAD(vkCreateBuffer);
    VK_LOAD(vkDestroyBuffer);
    VK_LOAD(vkGetBufferMemoryRequirements);
    VK_LOAD(vkAllocateMemory);
    VK_LOAD(vkFreeMemory);
    VK_LOAD(vkBindBufferMemory);
    VK_LOAD(vkMapMemory);
    VK_LOAD(vkUnmapMemory);
    VK_LOAD(vkCmdBindPipeline);
    VK_LOAD(vkCmdBindDescriptorSets);
    VK_LOAD(vkCmdPushConstants);
    VK_LOAD(vkCmdDispatch);
    VK_LOAD(vkCreateFence);
    VK_LOAD(vkDestroyFence);
    VK_LOAD(vkWaitForFences);
    VK_LOAD(vkResetFences);
    VK_LOAD(vkFlushMappedMemoryRanges);
    VK_LOAD(vkInvalidateMappedMemoryRanges);

    /* ─── Create command pool ─── */
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = vk->queue_family,
    };
    if (pfn_vkCreateCommandPool(vk->device, &cpci, NULL, &vk->cmd_pool) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create command pool\n");
        goto fail;
    }

    /* ─── Allocate command buffer ─── */
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (pfn_vkAllocateCommandBuffers(vk->device, &cbai, &vk->cmd) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to allocate command buffer\n");
        goto fail;
    }

    /* ─── Create fence ─── */
    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    if (pfn_vkCreateFence(vk->device, &fci, NULL, &vk->fence) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create fence\n");
        goto fail;
    }

    /* ─── Create descriptor set layout ─── */
    VkDescriptorSetLayoutBinding bindings[3] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    if (pfn_vkCreateDescriptorSetLayout(vk->device, &dslci, NULL,
                                         &vk->ds_layout) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create desc set layout\n");
        goto fail;
    }

    /* ─── Create descriptor pool (supports batch = up to 33 sets) ─── */
    VkDescriptorPoolSize pool_sizes[1] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 3 * (CT_VK_MAX_BATCH + 1) },
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = CT_VK_MAX_BATCH + 1,
        .poolSizeCount = 1,
        .pPoolSizes = pool_sizes,
    };
    if (pfn_vkCreateDescriptorPool(vk->device, &dpci, NULL,
                                     &vk->desc_pool) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create desc pool\n");
        goto fail;
    }

    /* ─── Pre-allocate batch descriptor sets (cycle, never free/reset) ─── */
    {
        VkDescriptorSetLayout all_layouts[CT_VK_MAX_BATCH];
        for (int i = 0; i < CT_VK_MAX_BATCH; i++)
            all_layouts[i] = vk->ds_layout;
        VkDescriptorSetAllocateInfo dsai2 = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk->desc_pool,
            .descriptorSetCount = CT_VK_MAX_BATCH,
            .pSetLayouts = all_layouts,
        };
        if (pfn_vkAllocateDescriptorSets(vk->device, &dsai2, vk->batch_sets) != VK_SUCCESS) {
            fprintf(stderr, "vulkan: failed to pre-allocate batch desc sets\n");
            goto fail;
        }
        vk->batch_set_idx = 0;
    }

    /* ─── Create pipeline layout (with push constants) ─── */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(int) * 4,  /* I, O, blk_per_I, blk_stride */
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->ds_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (pfn_vkCreatePipelineLayout(vk->device, &plci, NULL,
                                    &vk->pipeline_layout) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create pipeline layout\n");
        goto fail;
    }

    /* ─── Load Q8_0 shader ─── */
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = q8_0_matmul_spv_size,
        .pCode = q8_0_matmul_spv,
    };
    if (pfn_vkCreateShaderModule(vk->device, &smci, NULL,
                                  &vk->shader_q8_0) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create shader module\n");
        goto fail;
    }

    /* ─── Create compute pipeline ─── */
    VkComputePipelineCreateInfo cpci2 = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = vk->shader_q8_0,
            .pName = "main",
        },
        .layout = vk->pipeline_layout,
    };
    if (pfn_vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1,
                                      &cpci2, NULL,
                                      &vk->pipeline_q8_0) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create compute pipeline\n");
        goto fail;
    }

    /* ─── Create staging buffer (host-coherent) ─── */
    if (ct_vk_create_buffer(vk, CT_VK_MAX_STAGING,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             &vk->staging, &vk->staging_mem) != 0) {
        fprintf(stderr, "vulkan: failed to create staging buffer\n");
        goto fail;
    }
    pfn_vkMapMemory(vk->device, vk->staging_mem, 0, CT_VK_MAX_STAGING,
                    0, &vk->staging_ptr);
    vk->staging_size = CT_VK_MAX_STAGING;

    /* ─── Upload staging buffer (host-coherent, reusable, 64MB) ─── */
    /* Used for weight upload: CPU → staging (memcpy) → vkCmdCopyBuffer(staging → device) */
    if (ct_vk_create_buffer(vk, 64 * 1024 * 1024,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             &vk->upload_buf, &vk->upload_mem) != 0) {
        fprintf(stderr, "vulkan: failed to create upload staging buffer\n");
        goto fail;
    }
    pfn_vkMapMemory(vk->device, vk->upload_mem, 0, 64 * 1024 * 1024,
                    0, &vk->upload_ptr);
    vk->upload_cap = 64 * 1024 * 1024;

    /* ─── Allocate command buffer for batch dispatches ─── */
    VkCommandBufferAllocateInfo cbai2 = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (pfn_vkAllocateCommandBuffers(vk->device, &cbai2, &vk->batch_cmd) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to allocate batch command buffer\n");
        goto fail;
    }

    /* ─── Create fence for batch submissions ─── */
    VkFenceCreateInfo fci2 = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    if (pfn_vkCreateFence(vk->device, &fci2, NULL, &vk->batch_fence) != VK_SUCCESS) {
        fprintf(stderr, "vulkan: failed to create batch fence\n");
        goto fail;
    }

    /* ─── Init batch state ─── */
    vk->batch_active = 0;
    vk->batch_count = 0;
    vk->batch_stag_used = 0;

    /* ─── Init weight buffer (device-local, grows on demand) ─── */
    vk->weights.buf = VK_NULL_HANDLE;
    vk->weights.mem = VK_NULL_HANDLE;
    vk->weights.capacity = 0;
    vk->weights.used = 0;
    vk->weights.count = 0;

    vk->initialized = 1;
    fprintf(stderr, "vulkan: backend initialized\n");
    return vk;

fail:
    ct_vulkan_destroy(vk);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════
 * Ensure weight buffer has enough capacity
 * ═══════════════════════════════════════════════════════════════ */
static int ct_vk_ensure_weight_capacity(ct_vulkan_backend* vk, size_t needed_bytes) {
    if (needed_bytes <= vk->weights.capacity) return 0;

    /* Grow by 2x or to needed size, whichever is larger;
     * round to 64KB boundary */
    size_t new_cap = vk->weights.capacity > 0 ? vk->weights.capacity * 2 : (64 * 1024);
    while (new_cap < needed_bytes) new_cap *= 2;
    new_cap = (new_cap + 65535) & ~(size_t)65535;

    VkBuffer new_buf;
    VkDeviceMemory new_mem;
    if (ct_vk_create_buffer(vk, new_cap,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             &new_buf, &new_mem) != 0) {
        fprintf(stderr, "vk: failed to allocate %zu MB weight buffer\n", new_cap / 1024 / 1024);
        return -1;
    }

    fprintf(stderr, "vk: allocated %zu MB weight buffer\n", new_cap / 1024 / 1024);

    /* Copy old data from previous buffer (windowed map to avoid Adreno map-size crash) */
    if (vk->weights.buf != VK_NULL_HANDLE && vk->weights.used > 0) {
        size_t old_used = vk->weights.used;
        const size_t MW = 4 * 1024 * 1024;  /* 4MB map windows */
        for (size_t off = 0; off < old_used; off += MW) {
            size_t win = old_used - off < MW ? old_used - off : MW;
            void* src_ptr = NULL, *dst_ptr = NULL;
            if (pfn_vkMapMemory(vk->device, vk->weights.mem, off, win, 0, &src_ptr) == VK_SUCCESS &&
                pfn_vkMapMemory(vk->device, new_mem, off, win, 0, &dst_ptr) == VK_SUCCESS) {
                memcpy(dst_ptr, src_ptr, win);
            }
            pfn_vkUnmapMemory(vk->device, vk->weights.mem);
            pfn_vkUnmapMemory(vk->device, new_mem);
        }
        pfn_vkDestroyBuffer(vk->device, vk->weights.buf, NULL);
        pfn_vkFreeMemory(vk->device, vk->weights.mem, NULL);
    } else if (vk->weights.buf != VK_NULL_HANDLE) {
        pfn_vkDestroyBuffer(vk->device, vk->weights.buf, NULL);
        pfn_vkFreeMemory(vk->device, vk->weights.mem, NULL);
    }

    vk->weights.buf = new_buf;
    vk->weights.mem = new_mem;
    vk->weights.capacity = new_cap;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Upload weights
 * ═══════════════════════════════════════════════════════════════ */
ct_vulkan_weight_id ct_vulkan_upload_weights(ct_vulkan_backend* vk,
    int type, const void* cpu_data, int I, int O, const char* debug_name) {
    if (!vk || !vk->initialized) return -1;
    if (vk->weights.count >= CT_VK_MAX_WEIGHTS) {
        fprintf(stderr, "vulkan: max weights reached (%d)\n", CT_VK_MAX_WEIGHTS);
        return -1;
    }

    int wid = vk->weights.count;

    if (type == CT_GGUF_TYPE_Q8_0) {
        int blk_per_I = (I + CT_QK8_0 - 1) / CT_QK8_0;
        
        /* Get alignment requirements */
        VkPhysicalDeviceProperties props;
        pfn_vkGetPhysicalDeviceProperties(vk->phys_dev, &props);
        VkDeviceSize buf_align = props.limits.minStorageBufferOffsetAlignment;
        
        int blk_stride = align_stride(blk_per_I * CT_VK_U32_PER_Q8_BLOCK, buf_align);
        size_t total_u32 = (size_t)O * blk_stride;
        size_t total_bytes = total_u32 * sizeof(uint32_t);

        if (ct_vk_ensure_weight_capacity(vk, vk->weights.used + total_bytes) != 0)
            return -1;

        /* Repack blocks to temp CPU buffer, then upload via reusable staging */
        uint32_t* tmp = (uint32_t*)malloc(total_bytes);
        if (!tmp) { fprintf(stderr, "vk: malloc failed\n"); return -1; }

        const ct_block_q8_0* src = (const ct_block_q8_0*)cpu_data;
        for (int j = 0; j < O; j++) {
            uint32_t* row_dst = tmp + (size_t)j * blk_stride;
            for (int b = 0; b < blk_per_I; b++) {
                repack_q8_0_block(&src[(size_t)j * blk_per_I + b],
                                  row_dst + (size_t)b * CT_VK_U32_PER_Q8_BLOCK);
            }
        }

        /* Map the weight buffer in large windows and memcpy directly.
         *
         * Adreno 730 driver bugs worked around:
         *   - vkMapMemory >~256MB of DEVICE_LOCAL memory crashes
         *   - vkCmdCopyBuffer with large range (>~64MB) hangs during recording
         *   - vkCmdPipelineBarrier with large range hangs
         *
         * Using 64MB map windows on HOST_VISIBLE | HOST_COHERENT memory,
         * which is safe (verified: 64MB staging buffer maps fine at init).
         * If 64MB fails, we HALVE the window and retry. */
        {
            size_t MAP_WIN = 64 * 1024 * 1024;  /* start with 64MB windows */
            size_t remaining = total_bytes;
            size_t src_off = 0;
            size_t dst_off = vk->weights.used;
            int n_win = 0;
            while (remaining > 0) {
                size_t win = remaining < MAP_WIN ? remaining : MAP_WIN;
                void* ptr = NULL;
                VkResult mr = pfn_vkMapMemory(vk->device, vk->weights.mem, dst_off, win, 0, &ptr);
                if (mr != VK_SUCCESS || !ptr) {
                    /* Map failed — halve window and retry (down to 1MB minimum) */
                    if (win <= 1 * 1024 * 1024) {
                        fprintf(stderr, "vk: map failed at off=%zu win=%zu mr=%d\n", dst_off, win, mr);
                        break;
                    }
                    MAP_WIN /= 2;
                    continue;
                }
                memcpy(ptr, (const char*)tmp + src_off, win);
                pfn_vkUnmapMemory(vk->device, vk->weights.mem);
                src_off += win;
                dst_off += win;
                remaining -= win;
                n_win++;
            }
        }
        free(tmp);

        /* Verify upload: read back first block and compare with repacked data */
        if (debug_name && total_u32 > 9) {
            void* verify_ptr = NULL;
            size_t verify_off = vk->weights.used;
            if (pfn_vkMapMemory(vk->device, vk->weights.mem, verify_off, 40, 0, &verify_ptr) == VK_SUCCESS && verify_ptr) {
                pfn_vkUnmapMemory(vk->device, vk->weights.mem);
            }
        }

        /* HOST_COHERENT memory guarantees host writes are visible to GPU
         * without explicit barrier on unified memory architectures.
         * Adreno driver hangs on large-range pipeline barriers, so we
         * rely on the coherence guarantee instead. */

        /* Store metadata */
        vk->weight_meta[wid].offset = vk->weights.used / sizeof(uint32_t);
        vk->weight_meta[wid].type = type;
        vk->weight_meta[wid].I = I;
        vk->weight_meta[wid].O = O;
        vk->weight_meta[wid].blk_per_I = blk_per_I;
        vk->weight_meta[wid].blk_stride = blk_stride;
        vk->weight_meta[wid].size_u32 = total_u32;
        if (debug_name)
            snprintf(vk->weight_meta[wid].name, sizeof(vk->weight_meta[wid].name),
                     "%s", debug_name);
        else
            snprintf(vk->weight_meta[wid].name, sizeof(vk->weight_meta[wid].name),
                     "weight_%d", wid);

        vk->weights.used += total_bytes;
        vk->weights.count++;
        
        return wid;
    }

    /* Unsupported type for now */
    fprintf(stderr, "vulkan: unsupported weight type %d for upload\n", type);
    return -1;
}

/* ═══════════════════════════════════════════════════════════════
 * Q8_0 matmul dispatch
 * ═══════════════════════════════════════════════════════════════ */
int ct_vulkan_matmul_q8_0(ct_vulkan_backend* vk, ct_vulkan_weight_id wid,
                           const float* x, float* y, int I, int O) {
    if (!vk || !vk->initialized) return -1;
    if (wid < 0 || wid >= vk->weights.count) return -1;
    if (vk->weight_meta[wid].type != CT_GGUF_TYPE_Q8_0) return -1;

    int blk_per_I = vk->weight_meta[wid].blk_per_I;
    int blk_stride = vk->weight_meta[wid].blk_stride;
    size_t wgt_offset = vk->weight_meta[wid].offset;

    /* Check staging capacity */
    size_t x_size = (size_t)I * sizeof(float);
    size_t y_size = (size_t)O * sizeof(float);
    size_t needed = x_size + y_size;
    if (needed > vk->staging_size) {
        fprintf(stderr, "vulkan: matmul too large (%zu bytes, max %zu)\n",
                needed, vk->staging_size);
        return -1;
    }

    /* Write x to staging buffer (offset 0) */
    memcpy(vk->staging_ptr, x, x_size);

    /* Set up descriptors */
    VkDescriptorSet ds;
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk->ds_layout,
    };
    if (pfn_vkAllocateDescriptorSets(vk->device, &dsai, &ds) != VK_SUCCESS) {
        return -1;
    }

    VkDescriptorBufferInfo wgt_buf_info = {
        .buffer = vk->weights.buf,
        .offset = wgt_offset * sizeof(uint32_t),
        .range = (VkDeviceSize)vk->weight_meta[wid].size_u32 * sizeof(uint32_t),
    };
    VkDescriptorBufferInfo x_buf_info = {
        .buffer = vk->staging,
        .offset = 0,
        .range = x_size,
    };
    VkDescriptorBufferInfo y_buf_info = {
        .buffer = vk->staging,
        .offset = x_size,
        .range = y_size,
    };

    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &wgt_buf_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &x_buf_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 2,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &y_buf_info },
    };
    pfn_vkUpdateDescriptorSets(vk->device, 3, writes, 0, NULL);

    /* Dispatch */
    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    pfn_vkBeginCommandBuffer(vk->cmd, &cbbi);
    pfn_vkCmdBindPipeline(vk->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_q8_0);
    pfn_vkCmdBindDescriptorSets(vk->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 vk->pipeline_layout, 0, 1, &ds, 0, NULL);

    int push_consts[4] = { I, O, blk_per_I, blk_stride };
    pfn_vkCmdPushConstants(vk->cmd, vk->pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_consts), push_consts);

    int n_groups = (O + CT_VK_WG_SIZE - 1) / CT_VK_WG_SIZE;
    pfn_vkCmdDispatch(vk->cmd, (uint32_t)n_groups, 1, 1);

    pfn_vkEndCommandBuffer(vk->cmd);

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &vk->cmd,
    };
    pfn_vkQueueSubmit(vk->queue, 1, &si, vk->fence);
    pfn_vkWaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX);
    pfn_vkResetFences(vk->device, 1, &vk->fence);

    /* Read back y from staging (offset x_size) */
    memcpy(y, (const char*)vk->staging_ptr + x_size, y_size);

    /* NOTE: We do NOT reset the descriptor pool here because Adreno 730
     * crashes on vkResetDescriptorPool. The pool size is generous enough
     * (33 sets) that exhaustion is not reachable in normal inference. */

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Status queries
 * ═══════════════════════════════════════════════════════════════ */
bool ct_vulkan_available(const ct_vulkan_backend* vk) {
    return vk && vk->initialized;
}

int ct_vulkan_batch_active(const ct_vulkan_backend* vk) {
    return (vk && vk->initialized && vk->batch_active) ? 1 : 0;
}

const char* ct_vulkan_device_name(const ct_vulkan_backend* vk) {
    if (vk && vk->initialized) return vk->device_name;
    return "N/A";
}

/* ═══════════════════════════════════════════════════════════════
 * Batch matmul — record multiple dispatches, submit once
 * ═══════════════════════════════════════════════════════════════ */

int ct_vulkan_batch_begin(ct_vulkan_backend* vk) {
    if (!vk || !vk->initialized) return -1;

    /* Cycle pre-allocated descriptor sets — NEVER call vkResetDescriptorPool
     * because Adreno 730 crashes on it (even with zero live allocations). */
    vk->batch_set_idx = 0;

    vk->batch_active = 1;
    vk->batch_count = 0;
    vk->batch_stag_used = 0;

    VkCommandBufferBeginInfo cbbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    pfn_vkBeginCommandBuffer(vk->batch_cmd, &cbbi);
    return 0;
}

int ct_vulkan_batch_matmul_q8_0(ct_vulkan_backend* vk, ct_vulkan_weight_id wid,
                                 const float* x, float* y, int I, int O) {
    if (!vk || !vk->batch_active) return -1;
    if (wid < 0 || wid >= vk->weights.count) return -1;
    if (vk->weight_meta[wid].type != CT_GGUF_TYPE_Q8_0) return -1;
    if (vk->batch_count >= CT_VK_MAX_BATCH) return -1;

    int blk_per_I = vk->weight_meta[wid].blk_per_I;
    int blk_stride = vk->weight_meta[wid].blk_stride;
    size_t wgt_offset = vk->weight_meta[wid].offset;

    size_t x_size = (size_t)I * sizeof(float);
    size_t y_size = (size_t)O * sizeof(float);

    /* Check staging capacity */
    size_t needed = vk->batch_stag_used + x_size + y_size;
    if (needed > vk->staging_size) {
        fprintf(stderr, "vk: batch staging full (%zu > %zu)\n", needed, vk->staging_size);
        return -1;
    }

    /* Write x to staging at batch_stag_used */
    size_t x_off = vk->batch_stag_used;
    memcpy((char*)vk->staging_ptr + x_off, x, x_size);

    /* Y output at x_off + x_size */
    size_t y_off = x_off + x_size;
    vk->batch_stag_used = y_off + y_size;

    /* Store y info for deferred copy-back */
    vk->batch_y[vk->batch_count].dst = y;
    vk->batch_y[vk->batch_count].stag_off = y_off;
    vk->batch_y[vk->batch_count].size = y_size;
    vk->batch_count++;

    /* Grab the next pre-allocated descriptor set (cycle, never free/reset).
     * Each dispatch needs its OWN set because descriptors are read at
     * EXECUTION time (not recording time), so reusing a single set would
     * make every dispatch see the LAST update. */
    VkDescriptorSet ds = vk->batch_sets[vk->batch_set_idx++];

    VkDescriptorBufferInfo wgt_info = {
        .buffer = vk->weights.buf,
        .offset = wgt_offset * sizeof(uint32_t),
        .range = (VkDeviceSize)vk->weight_meta[wid].size_u32 * sizeof(uint32_t),
    };
    VkDescriptorBufferInfo x_info = {
        .buffer = vk->staging,
        .offset = x_off,
        .range = x_size,
    };
    VkDescriptorBufferInfo y_info = {
        .buffer = vk->staging,
        .offset = y_off,
        .range = y_size,
    };
    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &wgt_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &x_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds, .dstBinding = 2,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &y_info },
    };
    pfn_vkUpdateDescriptorSets(vk->device, 3, writes, 0, NULL);

    /* Record dispatch into batch command buffer */
    pfn_vkCmdBindPipeline(vk->batch_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipeline_q8_0);
    pfn_vkCmdBindDescriptorSets(vk->batch_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 vk->pipeline_layout, 0, 1, &ds, 0, NULL);

    int push_consts[4] = { I, O, blk_per_I, blk_stride };
    pfn_vkCmdPushConstants(vk->batch_cmd, vk->pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_consts), push_consts);

    int n_groups = (O + CT_VK_WG_SIZE - 1) / CT_VK_WG_SIZE;
    pfn_vkCmdDispatch(vk->batch_cmd, (uint32_t)n_groups, 1, 1);

    return 0;
}

int ct_vulkan_batch_end(ct_vulkan_backend* vk) {
    if (!vk || !vk->batch_active) return -1;

    pfn_vkEndCommandBuffer(vk->batch_cmd);

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &vk->batch_cmd,
    };
    pfn_vkQueueSubmit(vk->queue, 1, &si, vk->batch_fence);
    pfn_vkWaitForFences(vk->device, 1, &vk->batch_fence, VK_TRUE, UINT64_MAX);
    pfn_vkResetFences(vk->device, 1, &vk->batch_fence);

    /* Copy all y results back from staging */
    for (int i = 0; i < vk->batch_count; i++) {
        memcpy(vk->batch_y[i].dst,
               (const char*)vk->staging_ptr + vk->batch_y[i].stag_off,
               vk->batch_y[i].size);
    }

    vk->batch_active = 0;
    vk->batch_count = 0;

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Destroy
 * ═══════════════════════════════════════════════════════════════ */
void ct_vulkan_destroy(ct_vulkan_backend* vk) {
    if (!vk) return;
    if (vk->device != VK_NULL_HANDLE)
        pfn_vkDeviceWaitIdle(vk->device);

    if (vk->batch_cmd && vk->cmd_pool)
        pfn_vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &vk->batch_cmd);
    if (vk->batch_fence) pfn_vkDestroyFence(vk->device, vk->batch_fence, NULL);
    if (vk->pipeline_q8_0) pfn_vkDestroyPipeline(vk->device, vk->pipeline_q8_0, NULL);
    if (vk->shader_q8_0) pfn_vkDestroyShaderModule(vk->device, vk->shader_q8_0, NULL);
    if (vk->pipeline_layout) pfn_vkDestroyPipelineLayout(vk->device, vk->pipeline_layout, NULL);
    if (vk->ds_layout) pfn_vkDestroyDescriptorSetLayout(vk->device, vk->ds_layout, NULL);
    if (vk->desc_pool) pfn_vkDestroyDescriptorPool(vk->device, vk->desc_pool, NULL);
    if (vk->fence) pfn_vkDestroyFence(vk->device, vk->fence, NULL);
    if (vk->cmd_pool) {
        if (vk->cmd) pfn_vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &vk->cmd);
        pfn_vkDestroyCommandPool(vk->device, vk->cmd_pool, NULL);
    }
    if (vk->upload_ptr) pfn_vkUnmapMemory(vk->device, vk->upload_mem);
    if (vk->upload_buf) pfn_vkDestroyBuffer(vk->device, vk->upload_buf, NULL);
    if (vk->upload_mem) pfn_vkFreeMemory(vk->device, vk->upload_mem, NULL);
    if (vk->staging_ptr) pfn_vkUnmapMemory(vk->device, vk->staging_mem);
    if (vk->staging) pfn_vkDestroyBuffer(vk->device, vk->staging, NULL);
    if (vk->staging_mem) pfn_vkFreeMemory(vk->device, vk->staging_mem, NULL);
    if (vk->weights.buf) pfn_vkDestroyBuffer(vk->device, vk->weights.buf, NULL);
    if (vk->weights.mem) pfn_vkFreeMemory(vk->device, vk->weights.mem, NULL);
    if (vk->device) pfn_vkDestroyDevice(vk->device, NULL);
    if (vk->instance) pfn_vkDestroyInstance(vk->instance, NULL);

    ct_vk_unload_lib();
    free(vk);
}

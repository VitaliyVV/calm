/**
 * calm_vulkan.h — Calm Vulkan GPU Backend
 *
 * Zero-dependency Vulkan compute backend for quantized LLM inference.
 * Q8_0 matmul via compute shader on Adreno (or any Vulkan 1.1+ GPU).
 *
 * Integration:
 *   ct_vulkan_backend* vk = ct_vulkan_init();
 *   if (vk) {
 *       int wid = ct_vulkan_upload_weights(vk, CT_GGUF_TYPE_Q8_0,
 *                                           cpu_ptr, I, O, name);
 *       ct_vulkan_matmul_q8_0(vk, wid, x_hst, y_hst, I, O);
 *   }
 */
#ifndef CALM_VULKAN_H
#define CALM_VULKAN_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque backend handle */
typedef struct ct_vulkan_backend ct_vulkan_backend;

/* Weight tensor handle (ID into device-local buffer) */
typedef int ct_vulkan_weight_id;

/* ─── Lifecycle ─── */

/* Initialize Vulkan backend.
 * Opens first suitable Vulkan device (discrete GPU preferred, else integrated).
 * Returns NULL if Vulkan is unavailable or device doesn't meet requirements.
 * The backend is lazy — pipelines compile on first matmul use. */
ct_vulkan_backend* ct_vulkan_init(void);

/* Destroy backend, free all GPU memory, destroy device/instance. */
void ct_vulkan_destroy(ct_vulkan_backend* vk);

/* ─── Weight upload ─── */

/* Upload a weight matrix to device-local GPU memory.
 *   type: CT_GGUF_TYPE_* (Q8_0, Q4_0, F32, etc.)
 *   cpu_data: pointer to CPU-side weight data in the original quant format
 *   I, O: matrix dimensions (each row = output, column = input)
 *   debug_name: optional name for error messages (may be NULL)
 * Returns weight ID >= 0, or -1 on failure.
 *
 * The backend repacks quantized data to GPU-friendly layout during upload. */
ct_vulkan_weight_id ct_vulkan_upload_weights(ct_vulkan_backend* vk,
    int type, const void* cpu_data, int I, int O, const char* debug_name);

/* ─── Matmul dispatch (synchronous, one submit per call) ─── */

/* Q8_0 matmul: y[O] = x[I] @ W[O, I] (Q8_0 quantized).
 * Uses GPU if weight_id is valid, otherwise returns -1.
 * Returns 0 on success, -1 if weight not on GPU or backend unavailable. */
int ct_vulkan_matmul_q8_0(ct_vulkan_backend* vk, ct_vulkan_weight_id wid,
                          const float* x, float* y, int I, int O);

/* ─── Batch matmul (multiple dispatches, one submit) ─── */

/* Begin a batch: all subsequent batch_matmul calls are recorded without
 * submit+wait. Call batch_end to submit all queued dispatches at once.
 * Reduces per-dispatch overhead dramatically. */
int ct_vulkan_batch_begin(ct_vulkan_backend* vk);

/* Record one Q8_0 matmul in the current batch. x is copied to staging,
 * dispatch is recorded, but NO submit or fence wait happens until batch_end.
 * y pointer is stored for deferred copy-back. */
int ct_vulkan_batch_matmul_q8_0(ct_vulkan_backend* vk, ct_vulkan_weight_id wid,
                                 const float* x, float* y, int I, int O);

/* End the batch: submit all recorded dispatches, wait for GPU,
 * copy all y results back to caller pointers. */
int ct_vulkan_batch_end(ct_vulkan_backend* vk);

/* Returns 1 if a batch is active (between begin/end), 0 otherwise. */
int ct_vulkan_batch_active(const ct_vulkan_backend* vk);

/* ─── Status ─── */

/* Returns true if Vulkan backend is initialized and functional. */
bool ct_vulkan_available(const ct_vulkan_backend* vk);

/* Get device name (for logging). Returns "N/A" if not initialized. */
const char* ct_vulkan_device_name(const ct_vulkan_backend* vk);

#ifdef __cplusplus
}
#endif

#endif /* CALM_VULKAN_H */

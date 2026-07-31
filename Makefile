# Calm — Universal Local LLM Runtime
# Build: make <target>
# Targets:
#   make              — ARM NEON + Vulkan (default, for phone)
#   make x86          — x86 AVX2 pure CPU (for laptop, no Vulkan)
#   make x86-vk       — x86 AVX2 + Vulkan (for laptop with GPU)
#   make calm_convert — standalone converter tool
#   make clean        — remove build artifacts
#
# Device: clang, aarch64 (ARMv8.2-a NEON+i8mm)
# Optional: Vulkan compute backend (requires vulkan-headers + shaderc)

CC = clang
CFLAGS_ARM = -march=armv8.2-a+dotprod+i8mm+fp16
CFLAGS = -O2 -std=c11 $(CFLAGS_ARM) -DCT_NEON -DCT_VULKAN
CFLAGS_X86 = -O2 -std=gnu11 -mavx2 -mfma -DCT_AVX2
CFLAGS_X86_VK = -O2 -std=gnu11 -mavx2 -mfma -DCT_AVX2 -DCT_VULKAN
LDFLAGS = -lm -lpthread

# Sources
QUANT_SRC = calm_quant.c
GGUF_SRC = calm_gguf.c
CONVERT_SRC = calm_convert.c
INFER_SRC = calm_infer.c
SSM_SRC = calm_ssm.c
MLA_SRC = calm_mla.c
TOKENIZER_SRC = calm_tokenizer.c
SERVER_SRC = calm_server.c
TOOLS_SRC = calm_tools.c
MAIN_SRC = calm.c
VK_SRC = calm_vulkan.c

# Objects by build variant
COMMON_OBJ = $(QUANT_SRC:.c=.o) $(GGUF_SRC:.c=.o) $(INFER_SRC:.c=.o) \
             $(SSM_SRC:.c=.o) $(MLA_SRC:.c=.o) $(TOKENIZER_SRC:.c=.o) \
             $(SERVER_SRC:.c=.o) $(TOOLS_SRC:.c=.o) $(MAIN_SRC:.c=.o)
VK_OBJ = $(VK_SRC:.c=.o)

# All objects (ARM NEON + Vulkan)
CALM_OBJS = $(COMMON_OBJ) $(VK_OBJ)

# SPIR-V shaders
SHADER_DIR = shaders
SHADER_SRC = $(SHADER_DIR)/q8_0_matmul.comp
SHADER_SPV = $(SHADER_DIR)/q8_0_matmul.spv
SHADER_HEADER = $(SHADER_DIR)/q8_0_matmul_spv.h

.PHONY: all clean x86 x86-vk shaders

all: calm calm_convert

# ---- SPIR-V shader compilation ----
$(SHADER_SPV): $(SHADER_SRC)
	glslangValidator -V -o $@ $<

$(SHADER_HEADER): $(SHADER_SPV)
	glslangValidator -V --variable-name q8_0_matmul_spv_data -o $@ $(SHADER_SRC)

shaders: $(SHADER_HEADER)

# ---- ARM NEON + Vulkan (phone) ----
calm: $(SHADER_HEADER) $(CALM_OBJS)
	$(CC) $(CFLAGS) -o $@ $(filter-out $(SHADER_HEADER),$^) $(LDFLAGS) -ldl
	@echo "Built: $@ (ARM NEON + Vulkan)"

# ---- x86 AVX2 pure CPU (no Vulkan, shaders not needed) ----
x86: CFLAGS = $(CFLAGS_X86)
x86: $(COMMON_OBJ)
	$(CC) $(CFLAGS_X86) -o calm $(COMMON_OBJ) $(LDFLAGS)
	@echo "Built: calm (x86 AVX2, no Vulkan)"

# ---- x86 AVX2 + Vulkan (laptop with GPU) ----
x86-vk: CFLAGS = $(CFLAGS_X86_VK)
x86-vk: shaders $(COMMON_OBJ:.o=_x86vk.o) $(VK_OBJ:.o=_x86vk.o)
	$(CC) $(CFLAGS_X86_VK) -o calm-x86-vk \
		$(COMMON_OBJ:.o=_x86vk.o) $(VK_OBJ:.o=_x86vk.o) $(LDFLAGS) -ldl
	@echo "Built: calm-x86-vk (x86 AVX2 + Vulkan)"

# ---- x86 + Vulkan: compile with x86 VK flags ----
$(COMMON_OBJ:.o=_x86vk.o): %_x86vk.o: %.c
	$(CC) $(CFLAGS_X86_VK) -c $< -o $@

$(VK_OBJ:.o=_x86vk.o): %_x86vk.o: %.c
	$(CC) $(CFLAGS_X86_VK) -c $< -o $@

# ---- Convert tool (ARM) ----
calm_convert: $(CONVERT_SRC:.c=.o) $(GGUF_SRC:.c=.o) $(QUANT_SRC:.c=.o)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

# ---- Convert tool (x86) ----
calm_convert-x86: CFLAGS = $(CFLAGS_X86)
calm_convert-x86: $(CONVERT_SRC:.c=.o) $(GGUF_SRC:.c=.o) $(QUANT_SRC:.c=.o)
	$(CC) $(CFLAGS_X86) -o calm_convert $^ $(LDFLAGS)
	@echo "Built: calm_convert (x86 AVX2, no Vulkan)"

# ---- Compile rules (ARM with Vulkan, default) ----
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Clean ----
clean:
	rm -f *.o calm calm_convert calm-x86-vk
	rm -f $(SHADER_SPV) $(SHADER_HEADER)

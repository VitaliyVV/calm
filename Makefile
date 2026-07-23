# Calm — Universal Local LLM Runtime
# Build: make <target>
# Device: clang, aarch64 (ARMv8.2-a NEON+i8mm)
# Optional: Vulkan compute backend (requires vulkan-headers + shaderc)

CC = clang
CFLAGS = -O2 -std=c11 -march=armv8.2-a+dotprod+i8mm+fp16 -DCT_NEON
CFLAGS_VK = -DCT_VULKAN
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

# Objects
QUANT_OBJ = $(QUANT_SRC:.c=.o)
GGUF_OBJ = $(GGUF_SRC:.c=.o)
CONVERT_OBJ = $(CONVERT_SRC:.c=.o)
INFER_OBJ = $(INFER_SRC:.c=.o)
SSM_OBJ = $(SSM_SRC:.c=.o)
MLA_OBJ = $(MLA_SRC:.c=.o)
TOKENIZER_OBJ = $(TOKENIZER_SRC:.c=.o)
SERVER_OBJ = $(SERVER_SRC:.c=.o)
TOOLS_OBJ = $(TOOLS_SRC:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)
VK_OBJ = $(VK_SRC:.c=.o)

CALM_OBJS = $(MAIN_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) $(INFER_OBJ) \
            $(SSM_OBJ) $(MLA_OBJ) $(TOKENIZER_OBJ) $(SERVER_OBJ) $(TOOLS_OBJ) $(VK_OBJ)

# x86 AVX2 build (desktop/server)
CFLAGS_X86 = -O2 -std=c11 -mavx2 -mfma -DCT_AVX2

# SPIR-V shaders
SHADER_DIR = shaders
SHADER_SRC = $(SHADER_DIR)/q8_0_matmul.comp
SHADER_SPV = $(SHADER_DIR)/q8_0_matmul.spv
SHADER_HEADER = $(SHADER_DIR)/q8_0_matmul_spv.h

.PHONY: all clean x86 shaders

all: calm calm_convert calm-cpu

# ---- SPIR-V shader compilation ----
$(SHADER_SPV): $(SHADER_SRC)
	glslangValidator -V -o $@ $<

$(SHADER_HEADER): $(SHADER_SPV)
	glslangValidator -V --variable-name q8_0_matmul_spv_data -o $@ $(SHADER_SRC)

shaders: $(SHADER_HEADER)

# ---- Main binary (ARM NEON) ----
calm: $(SHADER_HEADER) $(CALM_OBJS)
	$(CC) $(CFLAGS) -o $@ $(filter-out $(SHADER_HEADER),$^) $(LDFLAGS) -ldl
	@echo "Built: $@ (ARM NEON + Vulkan)"

# ---- Vulkan-enabled build (ARM NEON + Vulkan) ----
calm-vk: $(SHADER_HEADER) $(CALM_OBJS) $(VK_OBJ)
	$(CC) $(CFLAGS) -o $@ $(filter-out $(SHADER_HEADER),$^) $(LDFLAGS) -ldl
	@echo "Built: $@ (ARM NEON + Vulkan)"

# ---- x86 build ----
x86: shaders
	$(MAKE) calm CFLAGS="$(CFLAGS_X86)"
	@echo "Built: calm (x86 AVX2)"

# ---- Convert tool ----
calm_convert: $(CONVERT_OBJ) $(GGUF_OBJ) $(QUANT_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

# ---- CPU-only build (no Vulkan) ----
CALM_CPU_OBJS = $(MAIN_SRC:.c=.o.cpu) $(QUANT_SRC:.c=.o.cpu) $(GGUF_SRC:.c=.o.cpu) \
                $(INFER_SRC:.c=.o.cpu) $(SSM_SRC:.c=.o.cpu) $(MLA_SRC:.c=.o.cpu) \
                $(TOKENIZER_SRC:.c=.o.cpu) $(SERVER_SRC:.c=.o.cpu) $(TOOLS_SRC:.c=.o.cpu)
calm-cpu: $(SHADER_HEADER) $(CALM_CPU_OBJS)
	$(CC) $(CFLAGS) -o $@ $(CALM_CPU_OBJS) $(LDFLAGS)
	@echo "Built: $@ (ARM NEON, no Vulkan)"

%.o.cpu: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Compile rules ----
.c.o:
	$(CC) $(CFLAGS) -DCT_VULKAN -c $< -o $@

# ---- Clean ----
clean:
	rm -f *.o calm calm_convert calm-vk
	rm -f $(SHADER_SPV) $(SHADER_HEADER)

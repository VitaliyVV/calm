# Calm — Universal Local LLM Runtime
# Build: make <target>
# Device: clang, aarch64 (ARMv8.2-a NEON+i8mm)

CC = clang
CFLAGS = -O2 -std=c11 -march=armv8.2-a+dotprod+i8mm+fp16 -DCT_NEON
LDFLAGS = -lm -lpthread

# Sources
QUANT_SRC = calm_quant.c
GGUF_SRC = calm_gguf.c
CONVERT_SRC = calm_convert.c
INFER_SRC = calm_infer.c
TOKENIZER_SRC = calm_tokenizer.c
SERVER_SRC = calm_server.c
TOOLS_SRC = calm_tools.c
MAIN_SRC = calm.c

# Objects
QUANT_OBJ = $(QUANT_SRC:.c=.o)
GGUF_OBJ = $(GGUF_SRC:.c=.o)
CONVERT_OBJ = $(CONVERT_SRC:.c=.o)
INFER_OBJ = $(INFER_SRC:.c=.o)
TOKENIZER_OBJ = $(TOKENIZER_SRC:.c=.o)
SERVER_OBJ = $(SERVER_SRC:.c=.o)
TOOLS_OBJ = $(TOOLS_SRC:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)

.PHONY: all clean run serve

all: calm calm_convert

calm: $(MAIN_OBJ) $(QUANT_OBJ) $(GGUF_OBJ) $(INFER_OBJ) $(TOKENIZER_OBJ) $(SERVER_OBJ) $(TOOLS_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

calm_convert: $(CONVERT_OBJ) $(GGUF_OBJ) $(QUANT_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o calm calm_convert

# Quick run (auto-detect GGUF)
RUN_MODEL ?= $(wildcard *.gguf)
run: calm
	@if [ -z "$(RUN_MODEL)" ]; then \
		echo "No .gguf found. Set RUN_MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	./calm run $(RUN_MODEL)

# Quick serve (auto-detect GGUF)
SERVE_PORT ?= 8080
serve: calm
	@if [ -z "$(RUN_MODEL)" ]; then \
		echo "No .gguf found. Set RUN_MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	./calm serve $(RUN_MODEL) --port $(SERVE_PORT)

NVCC ?= nvcc

# Adjust to your GPU's compute capability, e.g. sm_80 (A100), sm_86 (RTX 30xx),
# sm_89 (RTX 40xx), sm_90 (H100). mma.sync.m16n8k16 with fp16 requires sm_75+.
ARCH ?= sm_80

NVCC_FLAGS = -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas=-v

TARGET = flashattn_test
SRC = flash_attn_v1.cu 

all: $(TARGET)

$(TARGET): $(SRC) helpers.cuh
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
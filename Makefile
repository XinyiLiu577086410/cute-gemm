NVCC       ?= nvcc
ARCH       ?= sm_90a
STD        := c++17
OPT        := -O3
NVCCFLAGS  := -gencode arch=compute_90a,code=sm_90a -std=$(STD) $(OPT) --expt-relaxed-constexpr
LDFLAGS    := -lcublas -lcudart

# --- DeepGEMM support ---
# Requires: pip install -e ./deepgemm  (installs deep_gemm + PyTorch)
# Enable:   make DEEPGEMM=1
ifdef DEEPGEMM
    NVCCFLAGS += -DHAS_DEEPGEMM
	NVCCFLAGS  := -gencode arch=compute_90a,code=sm_90a -std=$(STD) $(OPT) --expt-relaxed-constexpr -I cutlass/include
endif

TARGET := testbed
SRCS   := main.cpp user_gemm.cu
DEPS   := testbed.hpp data_utils.hpp timing.hpp verify.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS) $(LDFLAGS) -I cutlass/include

clean:
	rm -f $(TARGET)

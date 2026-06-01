NVCC       ?= nvcc
ARCH       ?= sm_90
STD        := c++17
OPT        := -O3
NVCCFLAGS  := -arch=$(ARCH) -std=$(STD) $(OPT) --expt-relaxed-constexpr
LDFLAGS    := -lcublas -lcudart

# --- DeepGEMM support ---
# Requires: pip install -e ./deepgemm  (installs deep_gemm + PyTorch)
# Enable:   make DEEPGEMM=1
ifdef DEEPGEMM
    NVCCFLAGS += -DHAS_DEEPGEMM
endif

TARGET := testbed
SRCS   := main.cpp user_gemm.cu
DEPS   := testbed.hpp data_utils.hpp timing.hpp verify.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)

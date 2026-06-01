NVCC       ?= nvcc
ARCH       ?= sm_90
STD        := c++17
OPT        := -O3
NVCCFLAGS  := -arch=$(ARCH) -std=$(STD) $(OPT) --expt-relaxed-constexpr
LDFLAGS    := -lcublas -lcudart

# --- DeepGEMM support (requires PyTorch + pip install -e ./deepgemm) ---
# Define DEEPGEMM=1 on the command line to enable, e.g.:
#   make DEEPGEMM=1
# Custom paths (overrides submodule defaults):
#   make DEEPGEMM=1 DEEPGEMM_INC=/path/to/include DEEPGEMM_LIB=/path/to/lib
ifdef DEEPGEMM
    DEEPGEMM_DIR  ?= deepgemm
    DEEPGEMM_INC  ?= $(DEEPGEMM_DIR)/deep_gemm/include
    NVCCFLAGS     += -I$(DEEPGEMM_INC) -DHAS_DEEPGEMM
    ifdef DEEPGEMM_LIB
        LDFLAGS += -L$(DEEPGEMM_LIB)
    endif
    # DeepGEMM uses JIT compilation; no -ldeepgemm static library.
    # To link a custom shared wrapper, add it via DEEPGEMM_LIB and LDFLAGS_EXTRA.
    ifdef DEEPGEMM_LDFLAGS
        LDFLAGS += $(DEEPGEMM_LDFLAGS)
    endif
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

NVCC       ?= nvcc
ARCH       ?= sm_90
STD        := c++17
OPT        := -O3
NVCCFLAGS  := -arch=$(ARCH) -std=$(STD) $(OPT) --expt-relaxed-constexpr
LDFLAGS    := -lcublas -lcudart

# --- DeepGEMM support ---
# Define DEEPGEMM=1 on the command line to enable, e.g.:
#   make DEEPGEMM=1 DEEPGEMM_LIB=/path/to/lib DEEPGEMM_INC=/path/to/include
ifdef DEEPGEMM
    ifdef DEEPGEMM_INC
        NVCCFLAGS += -I$(DEEPGEMM_INC)
    endif
    NVCCFLAGS += -DHAS_DEEPGEMM
    ifdef DEEPGEMM_LIB
        LDFLAGS += -L$(DEEPGEMM_LIB)
    endif
    LDFLAGS += -ldeepgemm
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

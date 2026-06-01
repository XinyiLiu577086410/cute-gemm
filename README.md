# CUDA BF16 GEMM Testbed

评测手写基于wmma的bf16 GEMM内核性能，与cuBLAS/DeepGEMM对比。

## 编译

```bash
# 默认SM90（需CUDA 12+，H100/B200等）
make

# 其他架构（如RTX 3090）
make ARCH=sm_86

# 指定CUDA路径
make NVCC=/usr/local/cuda-12.4/bin/nvcc

# 启用DeepGEMM
make DEEPGEMM=1 DEEPGEMM_LIB=/path/to/lib DEEPGEMM_INC=/path/to/include

# 清理
make clean
```

## 运行

```bash
# 默认4096x4096x4096
./testbed

# 自定义尺寸和参数
./testbed -M 2048 -N 2048 -K 2048 -warmup 2 -iter 20

# 仅测试用户内核（跳过baseline）
./testbed --skip-cublas --skip-deepgemm

# 调整正确性阈值（bf16大K需放宽，默认1.0）
./testbed -M 4096 -N 4096 -K 4096 --threshold 5.0
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-M <int>` | M 维度 | 4096 |
| `-N <int>` | N 维度 | 4096 |
| `-K <int>` | K 维度 | 4096 |
| `-warmup <int>` | 热身迭代数 | 1 |
| `-iter <int>` | 测量迭代数 | 10 |
| `--skip-cublas` | 跳过cuBLAS | — |
| `--skip-deepgemm` | 跳过DeepGEMM | — |
| `--skip-user` | 跳过用户内核 | — |
| `--threshold <f>` | 正确性阈值 | 1.0 |
| `-h` | 打印帮助 | — |

## 用户内核接口

将你的GEMM实现写入 `user_gemm.cu`，保持以下签名：

```cpp
extern "C" void user_gemm(const __nv_bfloat16* dA,
                          const __nv_bfloat16* dB,
                          __nv_bfloat16* dC,
                          int M, int N, int K);
```

矩阵均为列主序（column-major），LDA=M, LDB=K, LDC=M。A为M×K，B为K×N，C为M×N。当前实现为64×64×64 wmma内核（仅支持M/N/K整除64）。

## 输出示例

```
================================================================================
  GEMM Benchmark Testbed
================================================================================
  Configuration: M=2048, N=2048, K=2048
  Device: NVIDIA GeForce RTX 3090
  SMs: 82, Clock: 1695 MHz
  Theoretical Peak BF16: 284651.5 GFLOPS
================================================================================

  Benchmark Results:
  Name                Time(ms)        GFLOPS      BW(GB/s)      Util(%)
  ----------------------------------------------------------------------
  cuBLAS                0.2892      59398.89         87.01         20.9%
  User GEMM             3.0250       5679.33          8.32          2.0%

  Speedup:
    User GEMM vs cuBLAS:   0.10x

  Correctness (User GEMM vs cuBLAS):
    Max Absolute Error: 5.00e-01
    Max Relative Error: 1.91e+03
    Threshold: 1.00
    Result: PASS
================================================================================
```

## 正确性阈值说明

bf16精度约3位有效数字。当K较大（≥1792）时，wmma与cuBLAS的累加顺序不同会导致舍入差异。典型max absolute error ≤ 0.5（K=2048时）。默认阈值1.0对该误差范围有效；可用 `--threshold` 调整。

## 文件结构

```
├── main.cpp          # CLI解析、benchmark流程、结果输出
├── testbed.hpp       # 类型定义、配置、设备信息、指标计算
├── data_utils.hpp    # CUDA/CUBLAS错误宏、数据生成、显存管理
├── timing.hpp        # cudaEvent计时、热身、离群值剔除
├── verify.hpp        # 逐元素正确性校验
├── user_gemm.cu      # 用户内核（64×64×64 wmma实现）
└── Makefile          # 编译系统
```

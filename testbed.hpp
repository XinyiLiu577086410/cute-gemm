#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <sstream>

// --- numerical constants with physical meaning ---
constexpr double kFMAFlops           = 2.0;
constexpr double kKhzToHz            = 1000.0;
constexpr double kMsToSec            = 1000.0;
constexpr double kGigaScale          = 1e9;
constexpr double kPctFactor          = 100.0;
constexpr double kDdrFactor          = 2.0;
constexpr double kBitsPerByte        = 8.0;

// dense bf16 FLOPS per SM per cycle (no 1:2 sparsity), per architecture
inline constexpr double kDenseBf16FlopsPerSmPerCycle(int sm_major) {
    if (sm_major >= 9) return 4096.0;  // Hopper
    if (sm_major >= 8) return 2048.0;  // Ampere
    return 512.0;                       // Volta/Turing
}

extern "C" void user_gemm(const __nv_bfloat16* dA,
                          const __nv_bfloat16* dB,
                          __nv_bfloat16* dC,
                          int M, int N, int K);

using GemmFunc = decltype(&user_gemm);

struct TestConfig {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int warmup = 1;
    int iterations = 10;
    bool skip_cublas = false;
    bool skip_deepgemm = false;
    bool skip_user = false;
    float error_threshold = 1.0f;
};

struct KernelResult {
    std::string name;
    double time_ms = 0.0;
    double gflops = 0.0;
    double bandwidth_gbs = 0.0;
    double compute_util_pct = 0.0;
    double bandwidth_util_pct = 0.0;
};

struct CorrectnessResult {
    bool passed = false;
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
};

struct DeviceInfo {
    int sm_count = 0;
    int clock_rate_khz = 0;
    int memory_clock_khz = 0;
    int memory_bus_width = 0;
    double theoretical_peak_gflops = 0.0;
    double theoretical_peak_bandwidth_gbs = 0.0;
    std::string name;
};

inline DeviceInfo query_device_info(int device_id = 0) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
                                 cudaGetErrorString(err));
    }
    DeviceInfo info;
    info.name = prop.name;
    info.sm_count = prop.multiProcessorCount;
    cudaDeviceGetAttribute(&info.clock_rate_khz,
                           cudaDevAttrClockRate, device_id);
    cudaDeviceGetAttribute(&info.memory_clock_khz,
                           cudaDevAttrMemoryClockRate, device_id);
    cudaDeviceGetAttribute(&info.memory_bus_width,
                           cudaDevAttrGlobalMemoryBusWidth, device_id);
    double clock_rate_hz = static_cast<double>(info.clock_rate_khz) * kKhzToHz;
    info.theoretical_peak_gflops =
        static_cast<double>(info.sm_count) * clock_rate_hz
        * kDenseBf16FlopsPerSmPerCycle(prop.major) / kGigaScale;
    double mem_clock_hz = static_cast<double>(info.memory_clock_khz) * kKhzToHz;
    info.theoretical_peak_bandwidth_gbs =
        mem_clock_hz * (static_cast<double>(info.memory_bus_width) / kBitsPerByte)
        * kDdrFactor / kGigaScale;
    return info;
}

inline double compute_gflops(int M, int N, int K, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    return (kFMAFlops * static_cast<double>(M) * static_cast<double>(N)
            * static_cast<double>(K)) / (time_ms / kMsToSec) / kGigaScale;
}

inline double compute_bandwidth_gbs(int M, int N, int K, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double bytes = (static_cast<double>(M) * K
                    + static_cast<double>(K) * N
                    + static_cast<double>(M) * N)
                   * static_cast<double>(sizeof(__nv_bfloat16));
    return bytes / (time_ms / kMsToSec) / kGigaScale;
}

inline double compute_utilization(double measured_gflops,
                                   double theoretical_peak_gflops) {
    if (theoretical_peak_gflops <= 0.0) return 0.0;
    return (measured_gflops / theoretical_peak_gflops) * kPctFactor;
}

inline double compute_bandwidth_utilization(double measured_bw_gbs,
                                              double theoretical_peak_bw_gbs) {
    if (theoretical_peak_bw_gbs <= 0.0) return 0.0;
    return (measured_bw_gbs / theoretical_peak_bw_gbs) * kPctFactor;
}

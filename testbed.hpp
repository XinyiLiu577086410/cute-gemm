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
    info.clock_rate_khz = prop.clockRate;
    double clock_rate_hz = static_cast<double>(info.clock_rate_khz) * 1000.0;
    info.theoretical_peak_gflops =
        static_cast<double>(info.sm_count) * clock_rate_hz * 1024.0 / 1e9;
    double mem_clock_hz = static_cast<double>(prop.memoryClockRate) * 1000.0;
    info.theoretical_peak_bandwidth_gbs =
        mem_clock_hz * (prop.memoryBusWidth / 8.0) * 2.0 / 1e9;
    return info;
}

inline double compute_gflops(int M, int N, int K, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    return (2.0 * static_cast<double>(M) * static_cast<double>(N) *
            static_cast<double>(K)) / (time_ms / 1000.0) / 1e9;
}

inline double compute_bandwidth_gbs(int M, int N, int K, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double bytes = (static_cast<double>(M) * K +
                    static_cast<double>(K) * N +
                    static_cast<double>(M) * N) * 2.0;
    return bytes / (time_ms / 1000.0) / 1e9;
}

inline double compute_utilization(double measured_gflops,
                                   double theoretical_peak_gflops) {
    if (theoretical_peak_gflops <= 0.0) return 0.0;
    return (measured_gflops / theoretical_peak_gflops) * 100.0;
}

inline double compute_bandwidth_utilization(double measured_bw_gbs,
                                              double theoretical_peak_bw_gbs) {
    if (theoretical_peak_bw_gbs <= 0.0) return 0.0;
    return (measured_bw_gbs / theoretical_peak_bw_gbs) * 100.0;
}

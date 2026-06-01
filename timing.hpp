#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <functional>

inline double measure_kernel_time_ms(std::function<void()> kernel_launch,
                                       int warmup_iters,
                                       int measure_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        kernel_launch();
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sync error during warmup: "
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA warmup sync failed");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times_ms;
    times_ms.reserve(measure_iters);

    for (int i = 0; i < measure_iters; ++i) {
        cudaEventRecord(start);
        kernel_launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times_ms.push_back(elapsed_ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (times_ms.empty()) return 0.0;

    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean = sum / static_cast<double>(times_ms.size());

    double sq_sum = 0.0;
    for (float t : times_ms) {
        double diff = static_cast<double>(t) - mean;
        sq_sum += diff * diff;
    }
    double stddev = std::sqrt(sq_sum / static_cast<double>(times_ms.size()));

    std::vector<float> filtered;
    for (float t : times_ms) {
        if (std::abs(static_cast<double>(t) - mean) <= 3.0 * stddev) {
            filtered.push_back(t);
        }
    }

    if (filtered.empty()) return mean;

    double filtered_sum =
        std::accumulate(filtered.begin(), filtered.end(), 0.0);
    return filtered_sum / static_cast<double>(filtered.size());
}

// Version where preparation (e.g. memset) is done outside the timed region.
// prepare() is called and synced before each timed kernel_launch().
inline double measure_kernel_time_ms_prepare(
    std::function<void()> prepare,
    std::function<void()> kernel_launch,
    int measure_iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times_ms;
    times_ms.reserve(measure_iters);

    for (int i = 0; i < measure_iters; ++i) {
        if (prepare) {
            prepare();
            cudaDeviceSynchronize();
        }
        cudaEventRecord(start);
        kernel_launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times_ms.push_back(elapsed_ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (times_ms.empty()) return 0.0;

    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean = sum / static_cast<double>(times_ms.size());

    double sq_sum = 0.0;
    for (float t : times_ms) {
        double diff = static_cast<double>(t) - mean;
        sq_sum += diff * diff;
    }
    double stddev = std::sqrt(sq_sum / static_cast<double>(times_ms.size()));

    std::vector<float> filtered;
    for (float t : times_ms) {
        if (std::abs(static_cast<double>(t) - mean) <= 3.0 * stddev) {
            filtered.push_back(t);
        }
    }

    if (filtered.empty()) return mean;

    double filtered_sum =
        std::accumulate(filtered.begin(), filtered.end(), 0.0);
    return filtered_sum / static_cast<double>(filtered.size());
}

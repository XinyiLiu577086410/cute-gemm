#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <random>
#include <stdexcept>
#include <iostream>
#include <cstring>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::cerr << "CUDA ERROR at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(_err) << std::endl;        \
            throw std::runtime_error("CUDA API call failed");                   \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _status = (call);                                        \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "CUBLAS ERROR at " << __FILE__ << ":" << __LINE__      \
                      << " - status " << static_cast<int>(_status) << std::endl;\
            throw std::runtime_error("CUBLAS API call failed");                 \
        }                                                                       \
    } while (0)

inline std::vector<__nv_bfloat16> generate_random_bf16(size_t count,
                                                         unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = __float2bfloat16(dist(rng));
    }
    return data;
}

inline __nv_bfloat16* allocate_device(size_t count) {
    __nv_bfloat16* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(__nv_bfloat16)));
    return d_ptr;
}

inline void copy_to_device(__nv_bfloat16* d_dst,
                            const std::vector<__nv_bfloat16>& h_src) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src.data(),
                           h_src.size() * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
}

inline std::vector<__nv_bfloat16> copy_to_host(__nv_bfloat16* d_src,
                                                  size_t count) {
    std::vector<__nv_bfloat16> h_data(count);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_src,
                           count * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));
    return h_data;
}

inline void zero_device(__nv_bfloat16* d_ptr, size_t count) {
    CUDA_CHECK(cudaMemset(d_ptr, 0, count * sizeof(__nv_bfloat16)));
}

inline void free_device(__nv_bfloat16* d_ptr) {
    if (d_ptr) {
        cudaFree(d_ptr);
    }
}

struct DeviceBuffers {
    __nv_bfloat16* dA = nullptr;
    __nv_bfloat16* dB = nullptr;
    __nv_bfloat16* dC = nullptr;
    size_t size_A = 0;
    size_t size_B = 0;
    size_t size_C = 0;

    void allocate(int M, int N, int K) {
        size_A = static_cast<size_t>(M) * K;
        size_B = static_cast<size_t>(K) * N;
        size_C = static_cast<size_t>(M) * N;
        dA = allocate_device(size_A);
        dB = allocate_device(size_B);
        dC = allocate_device(size_C);
    }

    void release() {
        free_device(dA); dA = nullptr;
        free_device(dB); dB = nullptr;
        free_device(dC); dC = nullptr;
    }
};

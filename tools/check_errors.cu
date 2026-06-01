#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA err at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(e));                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t s = (call);                                              \
        if (s != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLAS err at %s:%d: %d\n", __FILE__, __LINE__,  \
                    (int)s);                                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

extern "C" void user_gemm(const __nv_bfloat16* dA, const __nv_bfloat16* dB,
                          __nv_bfloat16* dC, int M, int N, int K);

int main(int argc, char* argv[]) {
    int M = 2048, N = 2048, K = 2048;
    unsigned seedA = 42, seedB = 123;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-M") && i + 1 < argc) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-K") && i + 1 < argc) K = atoi(argv[++i]);
    }

    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
    std::vector<__nv_bfloat16> hA(szA), hB(szB);

    std::mt19937 rngA(seedA), rngB(seedB);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < szA; i++) hA[i] = __float2bfloat16(dist(rngA));
    for (size_t i = 0; i < szB; i++) hB[i] = __float2bfloat16(dist(rngB));

    __nv_bfloat16 *dA, *dB, *dC_cublas, *dC_user;
    CUDA_CHECK(cudaMalloc(&dA, szA * 2));
    CUDA_CHECK(cudaMalloc(&dB, szB * 2));
    CUDA_CHECK(cudaMalloc(&dC_cublas, szC * 2));
    CUDA_CHECK(cudaMalloc(&dC_user, szC * 2));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB * 2, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaMemset(dC_cublas, 0, szC * 2));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                               &alpha, dA, CUDA_R_16BF, M, dB, CUDA_R_16BF, K,
                               &beta, dC_cublas, CUDA_R_16BF, M,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaMemset(dC_user, 0, szC * 2));
    user_gemm(dA, dB, dC_user, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hC_ref(szC), hC_user(szC);
    CUDA_CHECK(
        cudaMemcpy(hC_ref.data(), dC_cublas, szC * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hC_user.data(), dC_user, szC * 2, cudaMemcpyDeviceToHost));

    float max_abs = 0.0f, max_rel = 0.0f;
    int cnt_01 = 0, cnt_05 = 0, cnt_1 = 0, cnt_10 = 0;

    for (size_t i = 0; i < szC; i++) {
        float c = __bfloat162float(hC_user[i]);
        float r = __bfloat162float(hC_ref[i]);
        float ae = fabsf(c - r);
        if (ae > max_abs) max_abs = ae;
        if (fabsf(r) > 1e-8f) {
            float re = ae / fabsf(r);
            if (re > max_rel) max_rel = re;
        }
        if (ae > 0.01) cnt_01++;
        if (ae > 0.5) cnt_05++;
        if (ae > 1.0) cnt_1++;
        if (ae > 10.0) cnt_10++;
    }

    printf("M=%d N=%d K=%d  elements=%zu\n", M, N, K, szC);
    printf("MaxAbsErr = %.6f\n", max_abs);
    printf("MaxRelErr = %.2f\n", max_rel);
    printf("AE > 0.01: %d (%.2f%%)\n", cnt_01, 100.0 * cnt_01 / szC);
    printf("AE > 0.50: %d (%.2f%%)\n", cnt_05, 100.0 * cnt_05 / szC);
    printf("AE > 1.00: %d (%.2f%%)\n", cnt_1, 100.0 * cnt_1 / szC);
    printf("AE > 10.0: %d (%.2f%%)\n", cnt_10, 100.0 * cnt_10 / szC);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_cublas));
    CUDA_CHECK(cudaFree(dC_user));
    return 0;
}

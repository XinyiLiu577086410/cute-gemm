#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define BLK_M 64
#define BLK_N 64
#define BLK_K 64
#define WARP_SIZE 32

__global__ void gemm_bf16_64x64x64_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int a_off_m = bx * BLK_M;
    int b_off_n = by * BLK_N;
    int c_off_m = bx * BLK_M;
    int c_off_n = by * BLK_N;

    __shared__ __nv_bfloat16 smem_A[BLK_M * BLK_K];
    __shared__ __nv_bfloat16 smem_B[BLK_K * BLK_N];
    __shared__ float        smem_C[BLK_M * BLK_N];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int warp_m  = warp_id / 2;
    int warp_n  = warp_id % 2;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];

    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    for (int k_tile = 0; k_tile < K; k_tile += BLK_K) {

        for (int i = tid; i < BLK_M * BLK_K; i += num_threads) {
            int m = i / BLK_K;
            int k = i % BLK_K;
            int a_row = a_off_m + m;
            int a_col = k_tile + k;
            if (a_row < M && a_col < K) {
                smem_A[i] = A[a_row + a_col * M];
            } else {
                smem_A[i] = __float2bfloat16(0.0f);
            }
        }

        for (int i = tid; i < BLK_K * BLK_N; i += num_threads) {
            int k = i / BLK_N;
            int n = i % BLK_N;
            int b_row = k_tile + k;
            int b_col = b_off_n + n;
            if (b_row < K && b_col < N) {
                smem_B[i] = B[b_row + b_col * K];
            } else {
                smem_B[i] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        for (int k_step = 0; k_step < BLK_K; k_step += 16) {
            for (int mi = 0; mi < 2; mi++) {
                for (int ni = 0; ni < 2; ni++) {
                    int a_m = warp_m * 32 + mi * 16;
                    int a_k = k_step;
                    wmma::load_matrix_sync(a_frag,
                                           &smem_A[a_m * BLK_K + a_k],
                                           BLK_K);

                    int b_k = k_step;
                    int b_n = warp_n * 32 + ni * 16;
                    wmma::load_matrix_sync(b_frag,
                                           &smem_B[b_k * BLK_N + b_n],
                                           BLK_N);

                    wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            int c_m = warp_m * 32 + mi * 16;
            int c_n = warp_n * 32 + ni * 16;
            wmma::store_matrix_sync(&smem_C[c_m * BLK_N + c_n],
                                    c_frag[mi][ni],
                                    BLK_N,
                                    wmma::mem_row_major);
        }
    }

    __syncthreads();

    for (int i = tid; i < BLK_M * BLK_N; i += num_threads) {
        int m = i / BLK_N;
        int n = i % BLK_N;
        int c_row = c_off_m + m;
        int c_col = c_off_n + n;
        if (c_row < M && c_col < N) {
            C[c_row + c_col * M] = __float2bfloat16(smem_C[i]);
        }
    }
}

extern "C" void user_gemm(const __nv_bfloat16* dA,
                          const __nv_bfloat16* dB,
                          __nv_bfloat16* dC,
                          int M, int N, int K) {
    if (M % BLK_M != 0 || N % BLK_N != 0 || K % BLK_K != 0) {
        fprintf(stderr, "[user_gemm] ERROR: M,N,K must be multiples of 64. "
                "Got M=%d N=%d K=%d\n", M, N, K);
        return;
    }

    dim3 block(128);
    dim3 grid(M / BLK_M, N / BLK_N);

    gemm_bf16_64x64x64_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[user_gemm] Kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

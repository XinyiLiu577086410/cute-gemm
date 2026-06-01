#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

constexpr int kTileM  = 64;
constexpr int kTileN  = 64;
constexpr int kTileK  = 64;
constexpr int kWmmaM  = 16;
constexpr int kWmmaN  = 16;
constexpr int kWmmaK  = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 2;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsM * kWarpsN;

__global__ __launch_bounds__(kThreadsPerBlock)
void gemm_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                      const __nv_bfloat16* __restrict__ B,
                      __nv_bfloat16* __restrict__ C,
                      int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int a_off_m = bx * kTileM;
    int b_off_n = by * kTileN;
    int c_off_m = bx * kTileM;
    int c_off_n = by * kTileN;

    __shared__ __nv_bfloat16 smem_A[kTileM * kTileK];
    __shared__ __nv_bfloat16 smem_B[kTileK * kTileN];
    __shared__ float        smem_C[kTileM * kTileN];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int warp_id = tid / kWarpSize;
    int warp_m  = warp_id / kWarpsN;
    int warp_n  = warp_id % kWarpsN;

    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                   __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                   __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                   float> c_frag[kWarpsM][kWarpsN];

    for (int mi = 0; mi < kWarpsM; mi++) {
        for (int ni = 0; ni < kWarpsN; ni++) {
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    for (int k_tile = 0; k_tile < K; k_tile += kTileK) {

        for (int i = tid; i < kTileM * kTileK; i += num_threads) {
            int m = i / kTileK;
            int k = i % kTileK;
            int a_row = a_off_m + m;
            int a_col = k_tile + k;
            if (a_row < M && a_col < K) {
                smem_A[i] = A[a_row + a_col * M];
            } else {
                smem_A[i] = __float2bfloat16(0.0f);
            }
        }

        for (int i = tid; i < kTileK * kTileN; i += num_threads) {
            int k = i / kTileN;
            int n = i % kTileN;
            int b_row = k_tile + k;
            int b_col = b_off_n + n;
            if (b_row < K && b_col < N) {
                smem_B[i] = B[b_row + b_col * K];
            } else {
                smem_B[i] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        for (int k_step = 0; k_step < kTileK; k_step += kWmmaK) {
            for (int mi = 0; mi < kWarpsM; mi++) {
                for (int ni = 0; ni < kWarpsN; ni++) {
                    int a_m = warp_m * (kTileM / kWarpsM) + mi * kWmmaM;
                    int a_k = k_step;
                    wmma::load_matrix_sync(a_frag,
                                           &smem_A[a_m * kTileK + a_k],
                                           kTileK);

                    int b_k = k_step;
                    int b_n = warp_n * (kTileN / kWarpsN) + ni * kWmmaN;
                    wmma::load_matrix_sync(b_frag,
                                           &smem_B[b_k * kTileN + b_n],
                                           kTileN);

                    wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag,
                                   c_frag[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    for (int mi = 0; mi < kWarpsM; mi++) {
        for (int ni = 0; ni < kWarpsN; ni++) {
            int c_m = warp_m * (kTileM / kWarpsM) + mi * kWmmaM;
            int c_n = warp_n * (kTileN / kWarpsN) + ni * kWmmaN;
            wmma::store_matrix_sync(&smem_C[c_m * kTileN + c_n],
                                    c_frag[mi][ni],
                                    kTileN,
                                    wmma::mem_row_major);
        }
    }

    __syncthreads();

    for (int i = tid; i < kTileM * kTileN; i += num_threads) {
        int m = i / kTileN;
        int n = i % kTileN;
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
    if (M % kTileM != 0 || N % kTileN != 0 || K % kTileK != 0) {
        fprintf(stderr, "[user_gemm] ERROR: M,N,K must be multiples of %d. "
                "Got M=%d N=%d K=%d\n", kTileM, M, N, K);
        return;
    }

    dim3 block(kThreadsPerBlock);
    dim3 grid(M / kTileM, N / kTileN);

    gemm_bf16_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[user_gemm] Kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

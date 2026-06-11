#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/barrier.h"


using namespace cute;

constexpr int kNumSMs    = 132;
constexpr int kClusterM  = 2;
constexpr int kNumClusters = kNumSMs / kClusterM;

template <typename ElementA,
          typename ElementB,
          typename SmemLayoutA,  // (M,K,P)
          typename SmemLayoutB>  // (N,K,P)
struct SharedStorage {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> A;
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> B;
    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};



template <int kTileM,
          int kTileN,
          int kTileK,
          int kWarpSize, 
          int kWarpsPerGroup,
          int kNumWarpGroups,
          int kThreads,
          int kNumStages,
          typename SmemLayoutA,
          typename SmemLayoutB,
          typename TiledCopyA,
          typename TiledCopyB,
          typename TensorTypeC,
          typename TiledMMA>
__global__ static
__launch_bounds__(kThreads, 1)
__cluster_dims__(2, 1, 1)
void
gemm_kernel(
    int M, int N, int K,
    CUTE_GRID_CONSTANT const TiledCopyA tma_a,
    CUTE_GRID_CONSTANT const TiledCopyB tma_b,
    TensorTypeC           mC,
    TiledMMA              mma
)
{
    const int lane_id = cutlass::canonical_lane_idx();
    const int warp_id = cutlass::canonical_warp_idx_sync();
    const int warpgroup_id = cutlass::canonical_warp_group_idx();
    int lane_predicate = cute::elect_one_sync();
    
    extern __shared__ char smem_buffer[];
    using SharedStorage = SharedStorage<cute::bfloat16_t, cute::bfloat16_t, SmemLayoutA, SmemLayoutB>;
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
    
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));

    auto cta_tiler = make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;

    constexpr int tma_transaction_bytes = int(sizeof(cute::bfloat16_t)) * kTileM * kTileK
                                        + int(sizeof(cute::bfloat16_t)) * kTileN * kTileK;

    uint16_t mcast_mask_b = (1 << kClusterM) - 1;
    int cta_in_cluster = blockIdx.x;
    uint32_t peer_cta = cta_in_cluster ^ 1;

    int num_m_tiles = M / kTileM;
    int num_n_tiles = N / kTileN;
    int num_m_pairs = num_m_tiles / kClusterM;
    int num_work_items = num_m_pairs * num_n_tiles;

    for (int work_id = blockIdx.y; work_id < num_work_items; work_id += kNumClusters) {
        int m_pair = work_id % num_m_pairs;
        int n_tile = work_id / num_m_pairs;
        int m_tile = m_pair * kClusterM + cta_in_cluster;
        auto cta_coord = make_coord(m_tile, n_tile, _);

        Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
        Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

        auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                            group_modes<0,2>(sA), group_modes<0,2>(gA));
        auto [tBgB, tBsB] = tma_partition(tma_b, cta_in_cluster, Layout<Int<kClusterM>>{},
                                            group_modes<0,2>(sB), group_modes<0,2>(gB));

        Tensor tCgC = thr_mma.partition_C(gC);
        Tensor tCrC = thr_mma.make_fragment_C(tCgC);
        clear(tCrC);

        int k_tile_count = size<1>(tAgA);
        int k_tile = 0;

        if ((warp_id == 0) && lane_predicate) {
            CUTE_UNROLL
            for (int pipe = 0; pipe < kNumStages; ++pipe) {
                ProducerBarType::init(&producer_mbar[pipe],   1);
                ConsumerBarType::init(&consumer_mbar[pipe], kWarpsPerGroup * kClusterM);
            }
            for (int pipe = 0; pipe < kNumStages; ++pipe) {
                ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
                copy(tma_b.with(producer_mbar[pipe], mcast_mask_b), tBgB(_,k_tile), tBsB(_,pipe));
                --k_tile_count;
                ++k_tile;
            }
        }
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();

        auto write_state = cutlass::PipelineState<kNumStages>();
        auto read_state  = cutlass::PipelineState<kNumStages>();

        if (warpgroup_id == 0) {
            if ((warp_id == 0) && lane_predicate) {
                while(k_tile_count) {
                    int pipe = write_state.index();
                    ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
                    ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
                    copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
                    copy(tma_b.with(producer_mbar[pipe], mcast_mask_b), tBgB(_,k_tile), tBsB(_,pipe));
                    --k_tile_count;
                    ++write_state;
                    ++k_tile;
                }
            }
        } else if (warpgroup_id == 1)  {
            while(k_tile_count) {
                int read_pipe = read_state.index();
                ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
                warpgroup_arrive();
                gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (lane_id == 0) {
                    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
                    ConsumerBarType::arrive(&consumer_mbar[read_pipe], peer_cta, 1u);
                }
                --k_tile_count;
                ++read_state;
            }
            copy(tCrC, tCgC);
        }

        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    }
}




// column major
// NNN
extern "C" void user_gemm(const cute::bfloat16_t* dA, // N
                          const cute::bfloat16_t* dB, // N
                                cute::bfloat16_t* dC, // N
                          int M, 
                          int N, 
                          int K) 
{
    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 64;
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerGroup = 4;          // GMMA warp group = 4 warps
    constexpr int kNumWarpGroups = 2;          // 1 producer + 1 consumer
    constexpr int kThreadsPerWG = kWarpSize * kWarpsPerGroup;
    constexpr int kThreads = kThreadsPerWG * kNumWarpGroups;  // 256
    constexpr int kNumStages = 4;

    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<cute::bfloat16_t>{}, make_shape(Int<kTileM>{},Int<kTileK>{},Int<kNumStages>{}));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{}, make_shape(Int<kTileN>{},Int<kTileK>{},Int<kNumStages>{}));


    if (M % (kTileM * kClusterM) != 0 || N % kTileN != 0 || K % kTileK != 0) {
        fprintf(stderr, "[user_gemm] ERROR: M must be multiple of %d, N of %d, K of %d. "
                "Got M=%d N=%d K=%d\n", kTileM * kClusterM, kTileN, kTileK, M, N, K);
        return;
    }

    
    auto tensor_a = make_tensor(make_gmem_ptr(dA), make_layout(make_shape(M, K), make_stride(1, M)));
    auto tensor_b = make_tensor(make_gmem_ptr(dB), make_layout(make_shape(N, K), make_stride(K, 1)));
    auto tensor_c = make_tensor(make_gmem_ptr(dC), make_layout(make_shape(M, N), make_stride(1, M)));

    auto tma_load_a = make_tma_atom(
        SM90_TMA_LOAD{},
        tensor_a,
        sA(_,_,0),
        make_shape(Int<kTileM>{}, Int<kTileK>{})
    );

    auto tma_load_b = make_tma_atom(
        SM90_TMA_LOAD_MULTICAST{},
        tensor_b,
        sB(_,_,0),
        make_shape(Int<kTileN>{}, Int<kTileK>{}),
        Int<kClusterM>{}
    );

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::K>{});

    size_t smem_size = sizeof(
        SharedStorage<cute::bfloat16_t, cute::bfloat16_t, decltype(sA), decltype(sB)>);
    
    dim3 dimBlock(kThreads);
    dim3 dimGrid(kClusterM, kNumClusters, 1);

    auto kernel_ptr = reinterpret_cast<void const*>(
        &gemm_kernel<kTileM, kTileN, kTileK, kWarpSize, kWarpsPerGroup,
                    kNumWarpGroups, kThreads, kNumStages,
                    decltype(sA), decltype(sB),
                    decltype(tma_load_a), decltype(tma_load_b),
                    decltype(tensor_c), decltype(tiled_mma)>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));

    gemm_kernel<kTileM, 
                kTileN, 
                kTileK, 
                kWarpSize, 
                kWarpsPerGroup,
                kNumWarpGroups,
                kThreads,
                kNumStages,
                decltype(sA), 
                decltype(sB),
                decltype(tma_load_a), 
                decltype(tma_load_b), 
                decltype(tensor_c),
                decltype(tiled_mma)> <<<dimGrid, dimBlock, smem_size>>> (
        M, N, K, 
        tma_load_a,
        tma_load_b, 
        tensor_c,
        tiled_mma
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[user_gemm] Kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
using namespace cute;

template <typename ElementA,
          typename ElementB,
          typename SmemLayoutA,  // (M,K,P)
          typename SmemLayoutB>  // (N,K,P)
struct SharedStorage {
    // A 和 B 的 tile buffer (含 pipeline 维度)
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
void
gemm_kernel(
    int M, int N, int K,
    CUTE_GRID_CONSTANT const TiledCopyA tma_a,
    CUTE_GRID_CONSTANT const TiledCopyB tma_b,
    TensorTypeC           mC,
    TiledMMA              mma
)
{
    extern __shared__ char smem_buffer[];

    const int k_tiles = K / kTileK;
    const int t_id = threadIdx.x;
    const int lane_id = cutlass::canonical_lane_idx();
    const int warp_id = cutlass::canonical_warp_idx_sync();
    const int warpgroup_id = cutlass::canonical_warp_group_idx();
    int lane_predicate = cute::elect_one_sync();

    using SharedStorage = SharedStorage<cute::bfloat16_t, cute::bfloat16_t, SmemLayoutA, SmemLayoutB>;
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
    auto cta_tiler = make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    auto tma_slice_a = tma_a.get_slice(Int<0>{});
    auto tma_slice_b = tma_b.get_slice(Int<0>{});
    Tensor tAgA = tma_slice_a.partition_S(gA);
    Tensor tAsA = tma_slice_a.partition_D(sA);
    Tensor tBgB = tma_slice_b.partition_S(gB);
    Tensor tBsB = tma_slice_b.partition_D(sB);

    int k_tile_count = size<3>(tAgA);
    int k_tile = 0;

    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarType = cutlass::arch::ClusterBarrier;
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                        + sizeof(make_tensor_like(tensor<0>(tBsB)));
    // init pipeline
    if ((warp_id == 0) && lane_predicate) {
        CUTE_UNROLL
        for (int pipe = 0; pipe < kNumStages; ++pipe) {
            ProducerBarType::init(&producer_mbar[pipe],   1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }

        for (int pipe = 0; pipe < kNumStages; ++pipe) {
            // Set expected Tx Bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_,_,_,k_tile), tAsA(_,_,_,pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_,_,_,k_tile), tBsB(_,_,_,pipe));
            --k_tile_count;
            ++k_tile;
        }
    }
    __syncthreads();


    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1,X>{});
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    // Allocate accumulators and clear them
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
    clear(tCrC);

    // Allocate "fragments"
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)


    auto write_state = cutlass::PipelineState<kNumStages>();
    auto read_state  = cutlass::PipelineState<kNumStages>();

    if (warpgroup_id == 0) {
        // TMA warps
        if ((warp_id == 0) && lane_predicate) {
            while(k_tile_count) {
                int pipe = write_state.index();
                ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
                ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
                copy(tma_a.with(producer_mbar[pipe]), tAgA(_,_,_,k_tile), tAsA(_,_,_,pipe));
                copy(tma_b.with(producer_mbar[pipe]), tBgB(_,_,_,k_tile), tBsB(_,_,_,pipe));
                --k_tile_count;
                ++write_state;
                ++k_tile;
            }
        }
    } else if (warpgroup_id == 1)  {
        while(k_tile_count) {
            int read_pipe = read_state.index();
            ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
            // MMAs to cover 1 K_TILE
            warpgroup_arrive();
            gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
            warpgroup_commit_batch();
            // Wait for all MMAs in a K_TILE to complete
            warpgroup_wait<0>();
            // Notify that consumption is done
            ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
            --k_tile_count;
            ++read_state;
        }

        // store C
        copy(tCrC, tCgC);
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
    constexpr int kTileM = 64;
    constexpr int kTileN = 64;
    constexpr int kTileK = 16;
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerGroup = 4;          // GMMA warp group = 4 warps
    constexpr int kNumWarpGroups = 2;          // 1 producer + 1 consumer
    constexpr int kThreadsPerWG = kWarpSize * kWarpsPerGroup;
    constexpr int kThreads = kThreadsPerWG * kNumWarpGroups;  // 256
    constexpr int kNumStages = 2;

    // // A: (BLK_M, BLK_K, Stages), col-major 在 SMEM 中
    // using SmemLayoutA = Layout<
    //                         Shape<Int<kTileM>, Int<kTileK>, Int<kNumStages>>,
    //                         Stride<_1, Int<kTileM>, Int<kTileM * kTileK>>
    //                     >;

    // // B: (BLK_N, BLK_K, Stages), col-major 在 SMEM 中
    // using SmemLayoutB = Layout<
    //                         Shape<Int<kTileN>, Int<kTileK>, Int<kNumStages>>,
    //                         Stride<Int<kTileN>, _1, Int<kTileN * kTileK>>
    //                     >;
    
    auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<cute::bfloat16_t>{}, make_shape(Int<kTileM>{},Int<kTileK>{},Int<kNumStages>{}));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{}, make_shape(Int<kTileK>{},Int<kTileN>{},Int<kNumStages>{}));


    if (M % kTileM != 0 || N % kTileN != 0 || K % kTileK != 0) {
        fprintf(stderr, "[user_gemm] ERROR: M,N,K must be multiples of %d. "
                "Got M=%d N=%d K=%d\n", kTileM, M, N, K);
        return;
    }

    
    // 创建 TMA 描述符 (host 端)
    auto tensor_a = make_tensor(make_gmem_ptr(dA), make_layout(make_shape(M, K), make_stride(1, M)));
    auto tensor_b = make_tensor(make_gmem_ptr(dB), make_layout(make_shape(N, K), make_stride(K, 1)));
    auto tensor_c = make_tensor(make_gmem_ptr(dC), make_layout(make_shape(M, N), make_stride(1, M)));

    // A: (BLK_M, BLK_K)
    auto tma_load_a = make_tma_copy(
        SM90_TMA_LOAD{},
        tensor_a,
        sA(_,_,0)
    );

    // B: (BLK_N, BLK_K)
    auto tma_load_b = make_tma_copy(
        SM90_TMA_LOAD{},
        tensor_b,
        sB(_,_,0)
    );

    TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::MN,GMMA::Major::K>{});

    size_t smem_size = sizeof(
        SharedStorage<cute::bfloat16_t, cute::bfloat16_t, decltype(sA), decltype(sB)>);
    
    dim3 dimBlock(kThreads);
    dim3 dimGrid(M / kTileM, N / kTileN);

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

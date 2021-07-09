#ifndef CK_DRIVER_DYNAMIC_GEMM_XDLOPS_V2
#define CK_DRIVER_DYNAMIC_GEMM_XDLOPS_V2

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_xdlops_v2.hpp"
#include "gridwise_operation_wrapper.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AGlobalDesc,
          typename BGlobalDesc,
          typename CGlobalDesc,
          typename CBlockClusterDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPack,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadSliceLengths_K_M,
          typename ABlockTransferThreadClusterLengths_K_M,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_M,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K_N,
          typename BBlockTransferThreadClusterLengths_K_N,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_N,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGlobalIteratorHacks,
          typename BGlobalIteratorHacks,
          typename CGlobalIteratorHacks,
          typename AGlobalMoveSliceWindowIteratorHacks,
          typename BGlobalMoveSliceWindowIteratorHacks>
__host__ float launch_kernel_dynamic_gemm_xdlops_v2(const FloatAB* p_a_global,
                                                    const FloatAB* p_b_global,
                                                    FloatC* p_c_global,
                                                    const AGlobalDesc& a_k_m_global_desc,
                                                    const BGlobalDesc& b_k_n_global_desc,
                                                    const CGlobalDesc& c_m0_m1_n0_n1_global_desc,
                                                    const CBlockClusterDesc& c_block_cluster_desc,
                                                    AGlobalIteratorHacks,
                                                    BGlobalIteratorHacks,
                                                    CGlobalIteratorHacks,
                                                    AGlobalMoveSliceWindowIteratorHacks,
                                                    BGlobalMoveSliceWindowIteratorHacks,
                                                    index_t nrepeat)

{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto M = a_k_m_global_desc.GetLength(I1);
    const auto N = b_k_n_global_desc.GetLength(I1);
    const auto K = a_k_m_global_desc.GetLength(I0);

    if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    if(!(MPerBlock % MPerWave == 0 && NPerBlock % NPerWave == 0))
    {
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    // GEMM
    using gridwise_gemm =
        GridwiseDynamicGemm_km_kn_m0m1n0n1_xdlops_v2<BlockSize,
                                                     FloatAB,
                                                     FloatAcc,
                                                     FloatC,
                                                     CGlobalMemoryDataOperation,
                                                     AGlobalDesc,
                                                     BGlobalDesc,
                                                     CGlobalDesc,
                                                     CBlockClusterDesc,
                                                     MPerBlock,
                                                     NPerBlock,
                                                     KPerBlock,
                                                     MPerWave,
                                                     NPerWave,
                                                     KPack,
                                                     MRepeat,
                                                     NRepeat,
                                                     ABlockTransferThreadSliceLengths_K_M,
                                                     ABlockTransferThreadClusterLengths_K_M,
                                                     ABlockTransferThreadClusterArrangeOrder,
                                                     ABlockTransferSrcAccessOrder,
                                                     ABlockTransferSrcVectorDim,
                                                     ABlockTransferSrcScalarPerVector,
                                                     ABlockTransferDstScalarPerVector_M,
                                                     AThreadTransferSrcResetCoordinateAfterRun,
                                                     BBlockTransferThreadSliceLengths_K_N,
                                                     BBlockTransferThreadClusterLengths_K_N,
                                                     BBlockTransferThreadClusterArrangeOrder,
                                                     BBlockTransferSrcAccessOrder,
                                                     BBlockTransferSrcVectorDim,
                                                     BBlockTransferSrcScalarPerVector,
                                                     BBlockTransferDstScalarPerVector_N,
                                                     BThreadTransferSrcResetCoordinateAfterRun,
                                                     CThreadTransferSrcDstAccessOrder,
                                                     CThreadTransferSrcDstVectorDim,
                                                     CThreadTransferDstScalarPerVector,
                                                     AGlobalIteratorHacks,
                                                     BGlobalIteratorHacks,
                                                     CGlobalIteratorHacks,
                                                     AGlobalMoveSliceWindowIteratorHacks,
                                                     BGlobalMoveSliceWindowIteratorHacks>;

    const auto GridSize = (M / MPerBlock) * (N / NPerBlock);

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
    float ave_time = 0;

    const auto kernel = kernel_dynamic_gemm_xdlops_v2<gridwise_gemm,
                                                      FloatAB,
                                                      FloatAB,
                                                      FloatC,
                                                      remove_reference_t<AGlobalDesc>,
                                                      remove_reference_t<BGlobalDesc>,
                                                      remove_reference_t<CGlobalDesc>,
                                                      remove_reference_t<CBlockClusterDesc>>;

    ave_time = launch_and_time_kernel(kernel,
                                      nrepeat,
                                      dim3(GridSize),
                                      dim3(BlockSize),
                                      0,
                                      0,
                                      p_a_global,
                                      p_b_global,
                                      p_c_global,
                                      a_k_m_global_desc,
                                      b_k_n_global_desc,
                                      c_m0_m1_n0_n1_global_desc,
                                      c_block_cluster_desc);

    return ave_time;
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
    DeviceMem a_k_m_global_desc_device_buf(sizeof(AGlobalDesc));
    DeviceMem b_k_n_global_desc_device_buf(sizeof(BGlobalDesc));
    DeviceMem c_m0_m1_n0_n1_global_desc_device_buf(sizeof(CGlobalDesc));
    DeviceMem c_block_cluster_desc_device_buf(sizeof(c_block_cluster_desc));

    a_k_m_global_desc_device_buf.ToDevice(&a_k_m_global_desc);
    b_k_n_global_desc_device_buf.ToDevice(&b_k_n_global_desc);
    c_m0_m1_n0_n1_global_desc_device_buf.ToDevice(&c_m0_m1_n0_n1_global_desc);
    c_block_cluster_desc_device_buf.ToDevice(&c_block_cluster_desc);

    float ave_time = 0;

    const auto kernel = kernel_dynamic_gemm_xdlops_v1<gridwise_gemm,
                                                      FloatAB,
                                                      FloatAB,
                                                      FloatC,
                                                      remove_reference_t<AGlobalDesc>,
                                                      remove_reference_t<BGlobalDesc>,
                                                      remove_reference_t<CGlobalDesc>,
                                                      remove_reference_t<CBlockClusterDesc>>;

    ave_time = launch_and_time_kernel(
        kernel,
        nrepeat,
        dim3(GridSize),
        dim3(BlockSize),
        0,
        0,
        p_a_global,
        p_b_global,
        p_c_global,
        (void __CONSTANT__*)a_k_m_global_desc_device_buf.GetDeviceBuffer(),
        (void __CONSTANT__*)b_k_n_global_desc_device_buf.GetDeviceBuffer(),
        (void __CONSTANT__*)c_m0_m1_n0_n1_global_desc_device_buf.GetDeviceBuffer(),
        (void __CONSTANT__*)c_block_cluster_desc_device_buf.GetDeviceBuffer());

    return ave_time;
#endif
}

} // namespace ck
#endif

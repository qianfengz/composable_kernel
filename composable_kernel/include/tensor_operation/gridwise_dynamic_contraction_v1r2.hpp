#ifndef CK_GRIDWISE_DYNAMIC_CONTRACTION_V1R2_HPP
#define CK_GRIDWISE_DYNAMIC_CONTRACTION_V1R2_HPP

#include "common_header.hpp"
#include "dynamic_multi_index_transform_helper.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "blockwise_gemm_v2r2.hpp"
#include "blockwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_transfer.hpp"
#include "threadwise_dynamic_tensor_slice_set.hpp"

namespace ck {

template <typename GridwiseContraction,
          typename FloatAB,
          typename FloatC,
          typename AGK0GM0GM10GM11GK1GridDesc,
          typename BGK0GN0GN10GN11GK1GridDesc,
          typename CGM10BM0BM1GN10BN0BN1GridDesc,
          typename CBlockIdToGM10GN10BlockClusterAdaptor,
          bool HasMainKBlockLoop,
          bool HasDoubleTailKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_dynamic_contraction_v1r2(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AGK0GM0GM10GM11GK1GridDesc a_gk0_gm0_gm10_gm11_gk1_grid_desc,
            const BGK0GN0GN10GN11GK1GridDesc b_gk0_gn0_gn10_gn11_gk1_grid_desc,
            const CGM10BM0BM1GN10BN0BN1GridDesc c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
            const CBlockIdToGM10GN10BlockClusterAdaptor
                c_blockid_to_gm10_gn10_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseContraction::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    GridwiseContraction::Run(p_a_grid,
                             p_b_grid,
                             p_c_grid,
                             p_shared_block,
                             a_gk0_gm0_gm10_gm11_gk1_grid_desc,
                             b_gk0_gn0_gn10_gn11_gk1_grid_desc,
                             c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                             c_blockid_to_gm10_gn10_block_cluster_adaptor,
                             integral_constant<bool, HasMainKBlockLoop>{},
                             integral_constant<bool, HasDoubleTailKBlockLoop>{});
}

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AGK0GM0GM1GK1GridDesc,
          typename BGK0GN0GN1GK1GridDesc,
          typename CGM0GM1GN0GN1GridDesc,
          index_t GM1PerBlockGM11,
          index_t GN1PerBlockGN11,
          index_t KPerBlock,
          index_t M1PerThreadM111,
          index_t N1PerThreadN111,
          index_t KPerThread,
          index_t M11N11ThreadClusterM1100,
          index_t M11N11ThreadClusterN1100,
          index_t M11N11ThreadClusterM1101,
          index_t M11N11ThreadClusterN1101,
          typename ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks>
struct GridwiseDynamicContraction_k0m0m1k1_k0n0n1k1_m0m1n0n1_v1r2
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    // GM0 and GN0 need to known at compile-time
    static constexpr auto GM0 = CGM0GM1GN0GN1GridDesc{}.GetLength(I0);
    static constexpr auto GN0 = CGM0GM1GN0GN1GridDesc{}.GetLength(I2);
    static constexpr auto GK1 = AGK0GM0GM1GK1GridDesc{}.GetLength(I3);

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // lds max alignment
        // TODO: part of them should be moved into blockwise-gemm
        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = GK1;

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_gk0_gm0_gm10_gm11_gk1_block_desc =
            make_dynamic_naive_tensor_descriptor_aligned_v2(
                make_tuple(Number<KPerBlock>{}, GM0, I1, Number<GM1PerBlockGM11>{}, GK1),
                max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_gk0_gn0_gn10_gn11_gk1_block_desc =
            make_dynamic_naive_tensor_descriptor_aligned_v2(
                make_tuple(Number<KPerBlock>{}, GN0, I1, Number<GN1PerBlockGN11>{}, GK1),
                max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size = math::integer_least_multiple(
            a_gk0_gm0_gm10_gm11_gk1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size = math::integer_least_multiple(
            b_gk0_gn0_gn10_gn11_gk1_block_desc.GetElementSpaceSize(), max_lds_align);

        return 2 * (a_block_aligned_space_size + b_block_aligned_space_size) * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr bool
    CheckValidity(const AGK0GM0GM1GK1GridDesc& a_gk0_gm0_gm1_gk1_grid_desc,
                  const BGK0GN0GN1GK1GridDesc& b_gk0_gn0_gn1_gk1_grid_desc,
                  const CGM0GM1GN0GN1GridDesc& c_gm0_gm1_gn0_gn1_grid_desc)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(GM0)>>::value &&
                          is_known_at_compile_time<remove_cv_t<decltype(GN0)>>::value,
                      "wrong! GM0 and GN0 need to be known at compile-time");

        const auto GM1 = a_gk0_gm0_gm1_gk1_grid_desc.GetLength(I2);
        const auto GN1 = b_gk0_gn0_gn1_gk1_grid_desc.GetLength(I2);
        const auto GK0 = a_gk0_gm0_gm1_gk1_grid_desc.GetLength(I0);

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)

        return ((GM0 == c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I0) &&
                 GM1 == c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I1) &&
                 GN0 == c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I2) &&
                 GN1 == c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I3) &&
                 GM0 == a_gk0_gm0_gm1_gk1_grid_desc.GetLength(I1) &&
                 GM1 == a_gk0_gm0_gm1_gk1_grid_desc.GetLength(I2) &&
                 GN0 == b_gk0_gn0_gn1_gk1_grid_desc.GetLength(I1) &&
                 GN1 == b_gk0_gn0_gn1_gk1_grid_desc.GetLength(I2) &&
                 GK0 == b_gk0_gn0_gn1_gk1_grid_desc.GetLength(I0) &&
                 GK1 == b_gk0_gn0_gn1_gk1_grid_desc.GetLength(I3)) &&
                (GM1 % GM1PerBlockGM11 == 0 && GN1 % GN1PerBlockGN11 == 0 && GK0 % KPerBlock == 0));
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CGM0GM1GN0GN1GridDesc& c_gm0_gm1_gn0_gn1_grid_desc)
    {
        const auto GM1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I1);
        const auto GN1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I3);

        constexpr index_t GM11 = GM1PerBlockGM11;
        constexpr index_t GN11 = GN1PerBlockGN11;

        const index_t GM10 = GM1 / GM11;
        const index_t GN10 = GN1 / GN11;

        const index_t grid_size = GM10 * GN10;

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t GK0)
    {
        const bool has_main_k_block_loop = (GK0 + KPerBlock) / (2 * KPerBlock) > 1;

        return has_main_k_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasDoubleTailKBlockLoop(index_t GK0)
    {
        const bool has_double_tail_k_block_loop = (GK0 / KPerBlock) % 2 == 0;

        return has_double_tail_k_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeAGK0GM0GM10GM11GK1GridDescriptor(const AGK0GM0GM1GK1GridDesc& a_gk0_gm0_gm1_gk1_grid_desc)
    {
        const auto GK0 = a_gk0_gm0_gm1_gk1_grid_desc.GetLength(I0);
        const auto GM1 = a_gk0_gm0_gm1_gk1_grid_desc.GetLength(I2);

        const auto GM11 = Number<GM1PerBlockGM11>{};
        const auto GM10 = GM1 / GM11;

        const auto a_gk0_gm0_gm10_gm11_gk1_grid_desc = transform_dynamic_tensor_descriptor(
            a_gk0_gm0_gm1_gk1_grid_desc,
            make_tuple(make_pass_through_transform(GK0),
                       make_pass_through_transform(GM0),
                       make_unmerge_transform(make_tuple(GM10, GM11)),
                       make_pass_through_transform(GK1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}));

        return a_gk0_gm0_gm10_gm11_gk1_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeBGK0GN0GN10GN11GK1GridDescriptor(const BGK0GN0GN1GK1GridDesc& b_gk0_gn0_gn1_gk1_grid_desc)
    {
        const auto GK0 = b_gk0_gn0_gn1_gk1_grid_desc.GetLength(I0);
        const auto GN1 = b_gk0_gn0_gn1_gk1_grid_desc.GetLength(I2);

        const auto GN11 = Number<GN1PerBlockGN11>{};
        const auto GN10 = GN1 / GN11;

        const auto b_gk0_gn0_gn10_gn11_gk1_grid_desc = transform_dynamic_tensor_descriptor(
            b_gk0_gn0_gn1_gk1_grid_desc,
            make_tuple(make_pass_through_transform(GK0),
                       make_pass_through_transform(GN0),
                       make_unmerge_transform(make_tuple(GN10, GN11)),
                       make_pass_through_transform(GK1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}));

        return b_gk0_gn0_gn10_gn11_gk1_grid_desc;
    }

    __host__ __device__ static constexpr auto MakeCGM10BM0BM1GN10BN0BN1GridDescriptor(
        const CGM0GM1GN0GN1GridDesc& c_gm0_gm1_gn0_gn1_grid_desc)
    {
        const auto GM1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I1);
        const auto GN1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I3);

        constexpr auto GM11 = Number<GM1PerBlockGM11>{};
        constexpr auto GN11 = Number<GN1PerBlockGN11>{};

        const auto GM10 = GM1 / GM11;
        const auto GN10 = GN1 / GN11;

        constexpr auto BM = GM0 * GM11;
        constexpr auto BN = GN0 * GN11;

        constexpr auto BM1 =
            Number<M11N11ThreadClusterM1100 * M11N11ThreadClusterM1101 * M1PerThreadM111>{};
        constexpr auto BN1 =
            Number<M11N11ThreadClusterN1100 * M11N11ThreadClusterN1101 * N1PerThreadN111>{};

        constexpr auto BM0 = BM / BM1;
        constexpr auto BN0 = BN / BN1;

        const auto c_gm0_gm10_gm11_gn0_gn10_gn11_grid_desc = transform_dynamic_tensor_descriptor(
            c_gm0_gm1_gn0_gn1_grid_desc,
            make_tuple(make_pass_through_transform(GM0),
                       make_unmerge_transform(make_tuple(GM10, GM11)),
                       make_pass_through_transform(GN0),
                       make_unmerge_transform(make_tuple(GN10, GN11))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}));

        const auto c_gm10_bm_gn10_bn_grid_desc = transform_dynamic_tensor_descriptor(
            c_gm0_gm10_gm11_gn0_gn10_gn11_grid_desc,
            make_tuple(make_pass_through_transform(GM10),
                       make_merge_transform(make_tuple(GM0, GM11)),
                       make_pass_through_transform(GN10),
                       make_merge_transform(make_tuple(GN0, GN11))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}, Sequence<4>{}, Sequence<3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc = transform_dynamic_tensor_descriptor(
            c_gm10_bm_gn10_bn_grid_desc,
            make_tuple(make_pass_through_transform(GM10),
                       make_unmerge_transform(make_tuple(BM0, BM1)),
                       make_pass_through_transform(GN10),
                       make_unmerge_transform(make_tuple(BN0, BN1))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}));

        return c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc;
    }

    __host__ __device__ static constexpr auto MakeCBlockIdToGM10GN10BlockClusterAdaptor(
        const CGM0GM1GN0GN1GridDesc& c_gm0_gm1_gn0_gn1_grid_desc)
    {
        const auto GM1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I1);
        const auto GN1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I3);

        constexpr auto GM11 = Number<GM1PerBlockGM11>{};
        constexpr auto GN11 = Number<GN1PerBlockGN11>{};

        const auto GM10 = GM1 / GM11;
        const auto GN10 = GN1 / GN11;

        const auto c_blockid_to_gm10_gn10_block_cluster_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(GM10, GN10))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return c_blockid_to_gm10_gn10_block_cluster_adaptor;
    }

    using AGK0GM0GM10GM11GK1GridDesc =
        decltype(MakeAGK0GM0GM10GM11GK1GridDescriptor(AGK0GM0GM1GK1GridDesc{}));
    using BGK0GN0GN10GN11GK1GridDesc =
        decltype(MakeBGK0GN0GN10GN11GK1GridDescriptor(BGK0GN0GN1GK1GridDesc{}));
    using CGM10BM0BM1GN10BN0BN1GridDesc =
        decltype(MakeCGM10BM0BM1GN10BN0BN1GridDescriptor(CGM0GM1GN0GN1GridDesc{}));
    using CBlockIdToGM10GN10BlockClusterAdaptor =
        decltype(MakeCBlockIdToGM10GN10BlockClusterAdaptor(CGM0GM1GN0GN1GridDesc{}));

    template <bool HasMainKBlockLoop, bool HasDoubleTailKBlockLoop>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        FloatAB* __restrict__ p_shared_block,
        const AGK0GM0GM10GM11GK1GridDesc& a_gk0_gm0_gm10_gm11_gk1_grid_desc,
        const BGK0GN0GN10GN11GK1GridDesc& b_gk0_gn0_gn10_gn11_gk1_grid_desc,
        const CGM10BM0BM1GN10BN0BN1GridDesc& c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
        const CBlockIdToGM10GN10BlockClusterAdaptor& c_blockid_to_gm10_gn10_block_cluster_adaptor,
        integral_constant<bool, HasMainKBlockLoop>,
        integral_constant<bool, HasDoubleTailKBlockLoop>)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_a_grid, a_gk0_gm0_gm10_gm11_gk1_grid_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_b_grid, b_gk0_gn0_gn10_gn11_gk1_grid_desc.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpace::Global>(
            p_c_grid, c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc.GetElementSpaceSize());

        const auto GK0 = a_gk0_gm0_gm10_gm11_gk1_grid_desc.GetLength(I0);

        // divide block work by [GM10, GN10]
        const auto c_gm10_gn10_block_cluster_idx =
            c_blockid_to_gm10_gn10_block_cluster_adaptor.CalculateBottomIndex(
                make_multi_index(get_block_1d_id()));

        // HACK: this force index data into SGPR
        const index_t igm10 = __builtin_amdgcn_readfirstlane(c_gm10_gn10_block_cluster_idx[I0]);
        const index_t ign10 = __builtin_amdgcn_readfirstlane(c_gm10_gn10_block_cluster_idx[I1]);

        // lds max alignment
        // TODO: part of them should be moved into blockwise-gemm
        // TODO: change this. I think it needs multi-dimensional alignment
        constexpr auto max_lds_align = GK1;

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_gk0_gm0_gm10_gm11_gk1_block_desc =
            make_dynamic_naive_tensor_descriptor_aligned_v2(
                make_tuple(Number<KPerBlock>{}, GM0, I1, Number<GM1PerBlockGM11>{}, GK1),
                max_lds_align);

        // B matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto b_gk0_gn0_gn10_gn11_gk1_block_desc =
            make_dynamic_naive_tensor_descriptor_aligned_v2(
                make_tuple(Number<KPerBlock>{}, GN0, I1, Number<GN1PerBlockGN11>{}, GK1),
                max_lds_align);

        // A matrix in LDS memory for blockwise GEMM
        //   be careful of LDS alignment
        constexpr auto a_gk0_bm_gk1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, GM0 * Number<GM1PerBlockGM11>{}, GK1), max_lds_align);

        // B matrix in LDS memory for blockwise GEMM
        //   be careful of LDS alignment
        constexpr auto b_gk0_bn_gk1_block_desc = make_dynamic_naive_tensor_descriptor_aligned_v2(
            make_tuple(Number<KPerBlock>{}, GN0 * Number<GN1PerBlockGN11>{}, GK1), max_lds_align);

        static_assert(a_gk0_gm0_gm10_gm11_gk1_block_desc.GetElementSpaceSize() ==
                              a_gk0_bm_gk1_block_desc.GetElementSpaceSize() &&
                          b_gk0_gn0_gn10_gn11_gk1_block_desc.GetElementSpaceSize() ==
                              b_gk0_bn_gk1_block_desc.GetElementSpaceSize(),
                      "wrong!");

        // A matrix blockwise copy
        auto a_blockwise_copy = BlockwiseDynamicTensorSliceTransfer_v4r1<
            BlockSize,
            InMemoryDataOperation::Set,
            Sequence<KPerBlock, GM0, 1, GM1PerBlockGM11, GK1.value>,
            ABlockTransferThreadSliceLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferThreadClusterLengths_GK0_GM0_GM10_GM11_GK1,
            ABlockTransferThreadClusterArrangeOrder,
            FloatAB,
            FloatAB,
            decltype(a_gk0_gm0_gm10_gm11_gk1_grid_desc),
            decltype(a_gk0_gm0_gm10_gm11_gk1_block_desc),
            ABlockTransferSrcAccessOrder,
            Sequence<0, 1, 2, 3, 4>,
            ABlockTransferSrcVectorTensorLengths_GK0_GM0_GM10_GM11_GK1, // SrcVectorTensorLengths
            ABlockTransferDstVectorTensorLengths_GK0_GM0_GM10_GM11_GK1, // DstVectorTensorLengths
            ABlockTransferSrcVectorTensorContiguousDimOrder, // SrcVectorTensorContiguousDimOrder
            Sequence<0, 1, 2, 3, 4>,                         // DstVectorTensorContiguousDimOrder
            false,
            true>(a_gk0_gm0_gm10_gm11_gk1_grid_desc,
                  make_multi_index(0, 0, igm10, 0, 0),
                  a_gk0_gm0_gm10_gm11_gk1_block_desc,
                  make_multi_index(0, 0, 0, 0, 0));

        // B matrix blockwise copy
        auto b_blockwise_copy = BlockwiseDynamicTensorSliceTransfer_v4r1<
            BlockSize,
            InMemoryDataOperation::Set,
            Sequence<KPerBlock, GN0, 1, GN1PerBlockGN11, GK1.value>,
            BBlockTransferThreadSliceLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterLengths_GK0_GN0_GN10_GN11_GK1,
            BBlockTransferThreadClusterArrangeOrder,
            FloatAB,
            FloatAB,
            decltype(b_gk0_gn0_gn10_gn11_gk1_grid_desc),
            decltype(b_gk0_gn0_gn10_gn11_gk1_block_desc),
            BBlockTransferSrcAccessOrder,
            Sequence<0, 1, 2, 3, 4>,
            BBlockTransferSrcVectorTensorLengths_GK0_GN0_GN10_GN11_GK1, // SrcVectorTensorLengths
            BBlockTransferDstVectorTensorLengths_GK0_GN0_GN10_GN11_GK1, // DstVectorTensorLengths
            BBlockTransferSrcVectorTensorContiguousDimOrder, // SrcVectorTensorContiguousDimOrder
            Sequence<0, 1, 2, 3, 4>,                         // DstVectorTensorContiguousDimOrder
            false,
            true>(b_gk0_gn0_gn10_gn11_gk1_grid_desc,
                  make_multi_index(0, 0, ign10, 0, 0),
                  b_gk0_gn0_gn10_gn11_gk1_block_desc,
                  make_multi_index(0, 0, 0, 0, 0));

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[KPerBlock, GM1PerBlockGM11] is in LDS
        //     b_mtx[KPerBlocl, GN1PerBlockGN11] is in LDS
        //     c_mtx[GM1PerBlockGM11, GN1PerBlockGN11] is distributed among threads, and saved in
        //       register
        const auto blockwise_gemm =
            BlockwiseGemm_k0mk1_k0nk1_m0m1n0n1_v2r3_pipeline_2x2<BlockSize,
                                                                 FloatAB,
                                                                 FloatAB,
                                                                 FloatAcc,
                                                                 decltype(a_gk0_bm_gk1_block_desc),
                                                                 decltype(b_gk0_bn_gk1_block_desc),
                                                                 M1PerThreadM111,
                                                                 N1PerThreadN111,
                                                                 KPerThread,
                                                                 M11N11ThreadClusterM1100,
                                                                 M11N11ThreadClusterN1100,
                                                                 M11N11ThreadClusterM1101,
                                                                 M11N11ThreadClusterN1101,
                                                                 M1PerThreadM111,
                                                                 N1PerThreadN111>{};

        constexpr auto c_bm0_bm1_bn0_bn1_thread_tensor_lengths =
            decltype(blockwise_gemm)::GetCM0M1N0N1ThreadTensorLengths();

        constexpr auto c_bm0_bm1_bn0_bn1_thread_desc =
            make_dynamic_naive_tensor_descriptor_packed_v2(
                sequence_to_tuple_of_number(c_bm0_bm1_bn0_bn1_thread_tensor_lengths));

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_aligned_space_size = math::integer_least_multiple(
            a_gk0_gm0_gm10_gm11_gk1_block_desc.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_aligned_space_size = math::integer_least_multiple(
            b_gk0_gn0_gn10_gn11_gk1_block_desc.GetElementSpaceSize(), max_lds_align);

        FloatAB* p_a_block_double = p_shared_block;
        FloatAB* p_b_block_double = p_shared_block + 2 * a_block_aligned_space_size;

        // register allocation for output
        auto c_thread_buf = make_static_buffer<AddressSpace::Vgpr, FloatAcc>(
            c_bm0_bm1_bn0_bn1_thread_desc.GetElementSpaceSize());

        ThreadwiseDynamicTensorSliceSet_v1<FloatAcc,
                                           decltype(c_bm0_bm1_bn0_bn1_thread_desc),
                                           decltype(c_bm0_bm1_bn0_bn1_thread_tensor_lengths)>{}
            .Run(c_bm0_bm1_bn0_bn1_thread_desc,
                 make_tuple(I0, I0, I0, I0),
                 c_thread_buf,
                 FloatAcc{0});

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock, 0, 0, 0, 0);

        auto a_block_even_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_a_block_double, a_gk0_gm0_gm10_gm11_gk1_block_desc.GetElementSpaceSize());
        auto b_block_even_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_b_block_double, b_gk0_gn0_gn10_gn11_gk1_block_desc.GetElementSpaceSize());

        auto a_block_odd_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_a_block_double + a_block_aligned_space_size,
            a_gk0_gm0_gm10_gm11_gk1_block_desc.GetElementSpaceSize());
        auto b_block_odd_buf = make_dynamic_buffer<AddressSpace::Lds>(
            p_b_block_double + b_block_aligned_space_size,
            b_gk0_gn0_gn10_gn11_gk1_block_desc.GetElementSpaceSize());

        // LDS double buffer: preload data into LDS
        {
            a_blockwise_copy.RunRead(
                a_gk0_gm0_gm10_gm11_gk1_grid_desc, a_global_buf, AGridIteratorHacks{});
            b_blockwise_copy.RunRead(
                b_gk0_gn0_gn10_gn11_gk1_grid_desc, b_global_buf, BGridIteratorHacks{});

            a_blockwise_copy.RunWrite(a_gk0_gm0_gm10_gm11_gk1_block_desc, a_block_even_buf);
            b_blockwise_copy.RunWrite(b_gk0_gn0_gn10_gn11_gk1_block_desc, b_block_even_buf);
        }

        if constexpr(HasMainKBlockLoop)
        {
            index_t k_block_data_begin = 0;

            // LDS double buffer: main body
            // use Do-While loop instead of For loop to simplify control flow
            do
            {
                // even iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_gk0_gm0_gm10_gm11_gk1_grid_desc,
                                                    a_block_slice_copy_step,
                                                    AGridMoveSliceWindowIteratorHacks{});
                b_blockwise_copy.MoveSrcSliceWindow(b_gk0_gn0_gn10_gn11_gk1_grid_desc,
                                                    b_block_slice_copy_step,
                                                    BGridMoveSliceWindowIteratorHacks{});

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_gk0_gm0_gm10_gm11_gk1_grid_desc, a_global_buf, AGridIteratorHacks{});
                b_blockwise_copy.RunRead(
                    b_gk0_gn0_gn10_gn11_gk1_grid_desc, b_global_buf, BGridIteratorHacks{});

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(c_bm0_bm1_bn0_bn1_thread_desc,
                                   a_block_even_buf,
                                   b_block_even_buf,
                                   c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_gk0_gm0_gm10_gm11_gk1_block_desc, a_block_odd_buf);
                b_blockwise_copy.RunWrite(b_gk0_gn0_gn10_gn11_gk1_block_desc, b_block_odd_buf);

                // odd iteration
                a_blockwise_copy.MoveSrcSliceWindow(a_gk0_gm0_gm10_gm11_gk1_grid_desc,
                                                    a_block_slice_copy_step,
                                                    AGridMoveSliceWindowIteratorHacks{});
                b_blockwise_copy.MoveSrcSliceWindow(b_gk0_gn0_gn10_gn11_gk1_grid_desc,
                                                    b_block_slice_copy_step,
                                                    BGridMoveSliceWindowIteratorHacks{});

                __syncthreads();

                // LDS doubel buffer: load next data from device mem
                a_blockwise_copy.RunRead(
                    a_gk0_gm0_gm10_gm11_gk1_grid_desc, a_global_buf, AGridIteratorHacks{});
                b_blockwise_copy.RunRead(
                    b_gk0_gn0_gn10_gn11_gk1_grid_desc, b_global_buf, BGridIteratorHacks{});

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(
                    c_bm0_bm1_bn0_bn1_thread_desc, a_block_odd_buf, b_block_odd_buf, c_thread_buf);

                // LDS double buffer: store next data to LDS
                a_blockwise_copy.RunWrite(a_gk0_gm0_gm10_gm11_gk1_block_desc, a_block_even_buf);
                b_blockwise_copy.RunWrite(b_gk0_gn0_gn10_gn11_gk1_block_desc, b_block_even_buf);

                k_block_data_begin += 2 * KPerBlock;
            } while(k_block_data_begin < GK0 - 2 * KPerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailKBlockLoop) // if has 2 iteration left
        {
            a_blockwise_copy.MoveSrcSliceWindow(a_gk0_gm0_gm10_gm11_gk1_grid_desc,
                                                a_block_slice_copy_step,
                                                AGridMoveSliceWindowIteratorHacks{});
            b_blockwise_copy.MoveSrcSliceWindow(b_gk0_gn0_gn10_gn11_gk1_grid_desc,
                                                b_block_slice_copy_step,
                                                BGridMoveSliceWindowIteratorHacks{});

            __syncthreads();

            // LDS double buffer: load last data from device mem
            a_blockwise_copy.RunRead(
                a_gk0_gm0_gm10_gm11_gk1_grid_desc, a_global_buf, AGridIteratorHacks{});
            b_blockwise_copy.RunRead(
                b_gk0_gn0_gn10_gn11_gk1_grid_desc, b_global_buf, BGridIteratorHacks{});

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(
                c_bm0_bm1_bn0_bn1_thread_desc, a_block_even_buf, b_block_even_buf, c_thread_buf);

            // LDS double buffer: store last data to LDS
            a_blockwise_copy.RunWrite(a_gk0_gm0_gm10_gm11_gk1_block_desc, a_block_odd_buf);
            b_blockwise_copy.RunWrite(b_gk0_gn0_gn10_gn11_gk1_block_desc, b_block_odd_buf);

            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_bm0_bm1_bn0_bn1_thread_desc, a_block_odd_buf, b_block_odd_buf, c_thread_buf);
        }
        else // if has 1 iteration left
        {
            __syncthreads();

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(
                c_bm0_bm1_bn0_bn1_thread_desc, a_block_even_buf, b_block_even_buf, c_thread_buf);
        }

        // output: register to global memory
        {
            constexpr index_t M11 =
                M1PerThreadM111 * M11N11ThreadClusterM1100 * M11N11ThreadClusterM1101;
            constexpr index_t N11 =
                N1PerThreadN111 * M11N11ThreadClusterN1100 * M11N11ThreadClusterN1101;

            constexpr index_t M10 = GM1PerBlockGM11 / M11;
            constexpr index_t N10 = GN1PerBlockGN11 / N11;

            constexpr index_t M111 = M1PerThreadM111;
            constexpr index_t N111 = N1PerThreadN111;

            constexpr auto c_gm10_bm0_bm1_gn10_bn0_bn1_thread_desc =
                make_dynamic_naive_tensor_descriptor_packed_v2(
                    make_tuple(I1,
                               Number<c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I0]>{},
                               Number<c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I1]>{},
                               I1,
                               Number<c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I2]>{},
                               Number<c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I3]>{}));

            const auto c_bm0_bm1_bn0_bn1_thread_origin_idx_on_block =
                blockwise_gemm.CalculateCM0M1N0N1ThreadOriginOnBlock(get_thread_local_1d_id());

            ThreadwiseDynamicTensorSliceTransfer_v1r3<
                FloatAcc,
                FloatC,
                decltype(c_gm10_bm0_bm1_gn10_bn0_bn1_thread_desc),
                decltype(c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc),
                Sequence<1,
                         c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I0],
                         c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I1],
                         1,
                         c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I2],
                         c_bm0_bm1_bn0_bn1_thread_tensor_lengths[I3]>,
                CThreadTransferSrcDstAccessOrder,
                CThreadTransferSrcDstVectorDim,
                CThreadTransferDstScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>{c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                      make_multi_index(igm10,
                                       c_bm0_bm1_bn0_bn1_thread_origin_idx_on_block[I0],
                                       c_bm0_bm1_bn0_bn1_thread_origin_idx_on_block[I1],
                                       ign10,
                                       c_bm0_bm1_bn0_bn1_thread_origin_idx_on_block[I2],
                                       c_bm0_bm1_bn0_bn1_thread_origin_idx_on_block[I3])}
                .Run(c_gm10_bm0_bm1_gn10_bn0_bn1_thread_desc,
                     make_tuple(I0, I0, I0, I0, I0, I0),
                     c_thread_buf,
                     c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                     c_grid_buf,
                     CGridIteratorHacks{});
        }
    }
};

} // namespace ck
#endif

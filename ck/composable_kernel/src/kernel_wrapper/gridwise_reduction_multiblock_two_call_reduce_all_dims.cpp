/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "tensor_descriptor_helper.hpp"
#include "data_type_enum_helper.hpp"
#include "reduction_common.hpp"
#include "gridwise_2d_reduction_multiblock_second_call.hpp"
#include "gridwise_generic_reduction_wrapper_common.hpp"

using namespace ck;
using namespace wrapper;

using srcDataType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_DST_DATATYPE)>::type;
using compType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t blockSize = CK_PARAM_BLOCKSIZE; // tunable

constexpr index_t srcDims = CK_PARAM_IN_DIMS;

constexpr ReduceTensorOp_t reduceOp          = static_cast<ReduceTensorOp_t>(CK_PARAM_REDUCE_OP);
constexpr NanPropagation_t nanPropaOpt = CK_PARAM_NAN_PROPAGATE == 0
                                             ? NanPropagation_t::NOT_PROPAGATE_NAN
                                             : NanPropagation_t::PROPAGATE_NAN;
constexpr ReduceTensorIndices_t reduceIndicesOpt = CK_PARAM_REDUCE_INDICES == 0
                                                       ? ReduceTensorIndices_t::NO_INDICES
                                                       : ReduceTensorIndices_t::FLATTENED_INDICES;

constexpr bool indexable    = reduce_binary_operator<compType, reduceOp>::indexable;
constexpr bool need_indices = indexable && (reduceIndicesOpt != ReduceTensorIndices_t::NO_INDICES);

constexpr index_t dim0_thread_cluster_size = CK_PARAM_DIM0_THREAD_CLUSTER_SIZE;
constexpr index_t dim1_thread_cluster_size = CK_PARAM_DIM1_THREAD_CLUSTER_SIZE;
constexpr index_t dim0_thread_slice_size = CK_PARAM_DIM0_THREAD_SLICE_SIZE;
constexpr index_t dim1_thread_slice_size = CK_PARAM_DIM1_THREAD_SLICE_SIZE;
constexpr index_t vectorDim = CK_PARAM_VECTOR_DIM;
constexpr index_t vectorSize = CK_PARAM_VECTOR_SIZE;

// helper functions using variadic template arguments
template <index_t... Ns>
__device__ static auto make_tuple_from_array_and_index_seq(const int* lengths, Sequence<Ns...>)
{
    return make_tuple(static_cast<index_t>(lengths[Ns])...);
};

template <index_t arraySize>
__device__ static auto make_tuple_from_array(const int* lengths, Number<arraySize>)
{
    static_assert(arraySize >= 1 && arraySize <= 6, "The tensor should have 1 to 6 dimensions");

    constexpr auto index_seq = typename arithmetic_sequence_gen<0, arraySize, 1>::type{};

    return make_tuple_from_array_and_index_seq(lengths, index_seq);
};

extern "C" __global__ void gridwise_generic_reduce_1_prepare(int GridSize,
                                                             int BlkGroupSize,
                                                             int inLength0,
                                                             int inLength1,
                                                             int inLength2,
                                                             int inLength3,
                                                             int inLength4,
                                                             int inLength5,
                                                             int inStride0,
                                                             int inStride1,
                                                             int inStride2,
                                                             int inStride3,
                                                             int inStride4,
                                                             int inStride5,
                                                             void* __restrict__ ws_global)
{
    (void)GridSize;

    void* p_src2dDesc = ws_global;
    void* p_dst1dDesc = static_cast<char*>(ws_global) + 2048;

    const int srcLengths[6] = {inLength0, inLength1, inLength2, inLength3, inLength4, inLength5};
    const int srcStrides[6] = {inStride0, inStride1, inStride2, inStride3, inStride4, inStride5};

    const auto tupleSrcLengths = make_tuple_from_array(srcLengths, Number<srcDims>{});
    const auto tupleSrcStrides = make_tuple_from_array(srcStrides, Number<srcDims>{});
    const auto tupleDstLengths = make_tuple(1);
    const auto tupleDstStrides = make_tuple(1);

    const auto srcDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);
    const auto dstDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

    const auto one_dim_srcDesc = transform_tensor_descriptor(
            srcDesc,
            make_tuple(make_merge_transform(tupleSrcLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, srcDims, 1>::type{}),
            make_tuple(Sequence<0>{}));

    const auto src2dDesc =
            transform_tensor_descriptor(one_dim_srcDesc,
                                        make_tuple(make_unmerge_transform(
                                            make_tuple(1, one_dim_srcDesc.GetLength(Number<0>{})))),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0, 1>{}));

    constexpr int invariantLen = 1;
    const auto toReduceLen = src2dDesc.GetLength(Number<1>{});

    constexpr auto dim0_tile_size = dim0_thread_cluster_size * dim0_thread_slice_size;
    constexpr auto dim1_tile_size = dim1_thread_cluster_size * dim1_thread_slice_size;

    const index_t reduceSizePerBlock =
            (((toReduceLen + BlkGroupSize - 1) / BlkGroupSize + dim1_tile_size - 1) / dim1_tile_size) * dim1_tile_size;

    const auto srcPad1 = GridSize / BlkGroupSize * dim0_tile_size - invariantLen;
    const auto srcPad2 = reduceSizePerBlock * BlkGroupSize - toReduceLen;;

    auto src2dDesc_2 =
               transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_pad_transform(invariantLen, 0, srcPad1),
                                                       make_pad_transform(toReduceLen, 0, srcPad2)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));
    
    if(get_thread_local_1d_id() == 0)
        *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2;

    auto dst1dDesc = transform_tensor_descriptor(
            dstDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
            make_tuple(Sequence<0>{}));

    if(get_thread_local_1d_id() == 0)
        *static_cast<decltype(dst1dDesc)*>(p_dst1dDesc) = dst1dDesc;
};

template <index_t srcDims>
struct get_ref_desc_types
{
    using refType_one_dim_srcDesc = typename get_ref_1d_desc_types<srcDims>::refType_1dDesc;

    static constexpr auto ref_dstDesc = make_naive_tensor_descriptor(make_tuple(1), make_tuple(1));

    static constexpr auto ref_src2dDesc =
        transform_tensor_descriptor(refType_one_dim_srcDesc{},
                                    make_tuple(make_unmerge_transform(make_tuple(
                                        1, refType_one_dim_srcDesc{}.GetLength(Number<0>{})))),
                                    make_tuple(Sequence<0>{}),
                                    make_tuple(Sequence<0, 1>{}));

    static constexpr auto ref_invariantLen = ref_src2dDesc.GetLength(Number<0>{});
    static constexpr auto ref_toReduceLen  = ref_src2dDesc.GetLength(Number<1>{});

    using refType_src2dDesc_padded =
        decltype(transform_tensor_descriptor(ref_src2dDesc,
                                             make_tuple(make_pad_transform(ref_invariantLen, 0, 2),
                                                        make_pad_transform(ref_toReduceLen, 0, 2)),
                                             make_tuple(Sequence<0>{}, Sequence<1>{}),
                                             make_tuple(Sequence<0>{}, Sequence<1>{})));

    using refType_dst1dDesc_padded =
        decltype(transform_tensor_descriptor(ref_dstDesc,
                                             make_tuple(make_pad_transform(ref_invariantLen, 0, 2)),
                                             make_tuple(Sequence<0>{}),
                                             make_tuple(Sequence<0>{})));
};

using refType_src2dDesc_padded = typename get_ref_desc_types<srcDims>::refType_src2dDesc_padded;
using refType_dst1dDesc_padded = typename get_ref_desc_types<srcDims>::refType_dst1dDesc_padded;

extern "C" __global__ void gridwise_generic_reduce_1(int origReduceLen,
                                                     int BlkGroupSize,
                                                     float alpha,
                                                     const void* __restrict__ p_src_global,
                                                     float beta,
                                                     void* __restrict__ p_dst_global,
                                                     const void CONSTANT* ws_global,
                                                     long ws_buf2_bytes_offset,
                                                     void* __restrict__ indices_global)
{
    using opReduce = typename reduce_binary_operator<compType, reduceOp>::opType;
    using preUnaryOpType = typename reduce_unary_operator<compType, reduceOp, true, false>::preUnaryOp;
    using posUnaryOpType = typename reduce_unary_operator<compType, reduceOp, true, false>::posUnaryOp;

    (void)p_dst_global;
    (void)indices_global;

    const void* p_src2dDesc = cast_pointer_to_generic_address_space(ws_global);
    const void* p_dst1dDesc = static_cast<const char*>(p_src2dDesc) + 2048;
    void* ws_buf1_global    = const_cast<char*>(static_cast<const char*>(p_src2dDesc) + 4096);

    const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded*>(p_src2dDesc);
    const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded*>(p_dst1dDesc);

    preUnaryOpType preUnaryOp(origReduceLen);
    posUnaryOpType posUnaryOp(origReduceLen);

    using gridwise_2d_reduce = GridwiseReduction_xy_to_x_multiblock_two_call<srcDataType,
                                                                             dstDataType,
                                                                             compType,
                                                                             decltype(src2dDesc),
                                                                             decltype(dst1dDesc),
                                                                             opReduce,
                                                                             preUnaryOpType,
                                                                             posUnaryOpType,
                                                                             nanPropaOpt,
                                                                             blockSize,
                                                                             dim0_thread_cluster_size,
                                                                             dim1_thread_cluster_size,
                                                                             dim0_thread_slice_size,
                                                                             dim1_thread_slice_size,
                                                                             vectorDim,
                                                                             vectorSize>;

    void* const ws_buf2_global =
        ws_buf2_bytes_offset > 0
            ? static_cast<void*>(static_cast<char*>(ws_buf1_global) + ws_buf2_bytes_offset)
            : nullptr;

    if constexpr(need_indices)
        gridwise_2d_reduce::RunWithIndices(
                              src2dDesc,
                              dst1dDesc,
                              preUnaryOp,
			      posUnaryOp,
                              BlkGroupSize,
                              alpha,
                              static_cast<const srcDataType* const __restrict__>(p_src_global),
                              static_cast<compType* const __restrict__>(ws_buf1_global),
                              static_cast<int* const __restrict__>(ws_buf2_global));
    else
        gridwise_2d_reduce::Run(
                              src2dDesc,
                              dst1dDesc,
                              preUnaryOp,
                              posUnaryOp,
                              BlkGroupSize,
                              alpha,
                              static_cast<const srcDataType* const __restrict__>(p_src_global),
                              static_cast<compType* const __restrict__>(ws_buf1_global),
                              static_cast<int* const __restrict__>(ws_buf2_global));
};

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
#include "gridwise_generic_2d_reduction_blockwise.hpp"

using namespace ck;

using srcDataType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_DST_DATATYPE)>::type;
using compType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t blockSize = CK_PARAM_BLOCKSIZE; // tunable

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

extern "C" __global__ void
gridwise_generic_reduce_2_prepare(int GridSize, int BlkGroupSize, void* __restrict__ ws_global)
{
    (void)GridSize;

    void* p_src2dDesc = ws_global;
    void* p_dst1dDesc = static_cast<char*>(ws_global) + 2048;

    const auto tupleDstLengths = make_tuple(1);
    const auto tupleDstStrides = make_tuple(1);

    const auto dstDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

    const index_t invariantLen = 1;
    const index_t toReduceLen  = BlkGroupSize;

    const auto src2dDesc = make_naive_tensor_descriptor_packed(make_tuple(invariantLen, toReduceLen));

    constexpr auto dim0_tile_size = dim0_thread_cluster_size * dim0_thread_slice_size;
    constexpr auto dim1_tile_size = dim1_thread_cluster_size * dim1_thread_slice_size;

    const auto srcPad1 = GridSize * dim0_tile_size - invariantLen;
    const auto srcPad2 = ((toReduceLen + dim1_tile_size - 1) / dim1_tile_size) * dim1_tile_size - toReduceLen;

    auto src2dDesc_2 = transform_tensor_descriptor(src2dDesc,
                                            make_tuple(make_right_pad_transform(invariantLen, srcPad1),
                                                       make_right_pad_transform(toReduceLen, srcPad2)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));
    if(get_thread_local_1d_id() == 0)
         *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2;

    const auto dstPad = srcPad1;

    auto dst1dDesc_2 =
            transform_tensor_descriptor(dstdDesc,
                                        make_tuple(make_right_pad_transform(invariantLen, dstPad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));

    if(get_thread_local_1d_id() == 0)
        *static_cast<decltype(dst1dDesc_2)*>(p_dst1dDesc) = dst1dDesc_2;
};

struct get_ref_desc_types
{
    static constexpr auto ref_tupleDstLengths = make_tuple(8);
    static constexpr auto ref_dstDesc =
        make_naive_tensor_descriptor(ref_tupleDstLengths, ref_tupleDstLengths);

    static constexpr index_t ref_invariantLen = ref_dstDesc.GetLength(Number<0>{});
    static constexpr index_t ref_toReduceLen  = 8;

    static constexpr auto ref_src2dDesc =
        make_naive_tensor_descriptor_packed(make_tuple(ref_invariantLen, ref_toReduceLen));

    using refType_src2dDesc_padded =
        decltype(transform_tensor_descriptor(ref_src2dDesc,
                                             make_tuple(make_right_pad_transform(ref_invariantLen, 2),
                                                        make_right_pad_transform(ref_toReduceLen, 2)),
                                             make_tuple(Sequence<0>{}, Sequence<1>{}),
                                             make_tuple(Sequence<0>{}, Sequence<1>{})));

    using refType_dst1dDesc_padded =
        decltype(transform_tensor_descriptor(ref_dstDesc,
                                             make_tuple(make_right_pad_transform(ref_invariantLen, 2)),
                                             make_tuple(Sequence<0>{}),
                                             make_tuple(Sequence<0>{})));
};

using refType_src2dDesc_padded = typename get_ref_desc_types::refType_src2dDesc_padded;
using refType_dst1dDesc_padded = typename get_ref_desc_types::refType_dst1dDesc_padded;

extern "C" __global__ void gridwise_generic_reduce_2(int origReduceLen,
                                                     float alpha,
                                                     const void* __restrict__ p_src_global,
                                                     float beta,
                                                     void* __restrict__ p_dst_global,
                                                     const void CONSTANT* ws_global,
                                                     long ws_buf2_bytes_offset,
                                                     void* __restrict__ indices_global)
{
    using opReduce = typename reduce_binary_operator<compType, reduceOp>::opType;
    using preUnaryOpType = typename reduce_unary_operator<compType, reduceOp, false, true>::preUnaryOp;
    using posUnaryOpType = typename reduce_unary_operator<compType, reduceOp, false, true>::posUnaryOp;

    (void)p_src_global;

    const void* p_src2dDesc = cast_pointer_to_generic_address_space(ws_global);
    const void* p_dst1dDesc = static_cast<const char*>(p_src2dDesc) + 2048;
    void* ws_buf1_global    = const_cast<char*>(static_cast<const char*>(p_src2dDesc) + 4096);

    const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded*>(p_src2dDesc);
    const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded*>(p_dst1dDesc);

    preUnaryOpType preUnaryOp(origReduceLen);
    posUnaryOpType posUnaryOp(origReduceLen);

    using gridwise_2d_reduce = GridwiseReduction_xy_to_x_blockwisel<compType,
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
        gridwise_2d_reduce::RunSecondCallWithIndice(
                              src2dDesc,
                              dst1dDesc,
                              preUnaryOp,
			      posUnaryOp,
                              alpha,
                              static_cast<const compType* const __restrict__>(ws_buf1_global),
                              beta,
                              static_cast<dstDataType* const __restrict__>(p_dst_global),
                              static_cast<const int* const __restrict__>(ws_buf2_global),
                              static_cast<int* const __restrict__>(indices_global));
    else
        gridwise_2d_reduce::Run(
                              src2dDesc,
                              dst1dDesc,
                              preUnaryOp,
                              posUnaryOp,
                              alpha,
                              static_cast<const compType* const __restrict__>(ws_buf1_global),
                              beta,
                              static_cast<dstDataType* const __restrict__>(p_dst_global),
                              static_cast<const int* const __restrict__>(ws_buf2_global),
                              static_cast<int* const __restrict__>(indices_global));
};

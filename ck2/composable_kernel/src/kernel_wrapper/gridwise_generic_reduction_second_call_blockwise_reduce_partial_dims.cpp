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
#include "gridwise_generic_reduction_wrapper_common.hpp"

using namespace ck;
using namespace wrapper;

using srcDataType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_DST_DATATYPE)>::type;
using compType =
    typename get_datatype_from_enum<static_cast<DataTypeEnum_t>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t BlockSize = CK_PARAM_BLOCKSIZE; // tunable

constexpr index_t dstDims = CK_PARAM_OUT_DIMS;

constexpr ReduceTensorOp_t op          = static_cast<ReduceTensorOp_t>(CK_PARAM_REDUCE_OP);
constexpr NanPropagation_t nanPropaOpt = CK_PARAM_NAN_PROPAGATE == 0
                                             ? NanPropagation_t::NOT_PROPAGATE_NAN
                                             : NanPropagation_t::PROPAGATE_NAN;
constexpr ReduceTensorIndices_t reduceIndicesOpt = CK_PARAM_REDUCE_INDICES == 0
                                                       ? ReduceTensorIndices_t::NO_INDICES
                                                       : ReduceTensorIndices_t::FLATTENED_INDICES;

constexpr bool src2d_need_padding = static_cast<bool>(CK_PARAM_SRC2D_PADDING);
constexpr bool dst1d_need_padding = static_cast<bool>(CK_PARAM_DST1D_PADDING);

constexpr bool indexable    = reduce_binary_operator<compType, op>::indexable;
constexpr bool need_indices = indexable && (reduceIndicesOpt != ReduceTensorIndices_t::NO_INDICES);

constexpr index_t GredAccessesPerThreadInBlock = CK_PARAM_DIM1_THREAD_SLICE_LENGTH;

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

extern "C" __global__ void gridwise_generic_reduce_2_prepare(int GridSize,
                                                             int BlkGroupSize,
                                                             int outLength0,
                                                             int outLength1,
                                                             int outLength2,
                                                             int outLength3,
                                                             int outLength4,
                                                             int outLength5,
                                                             int outStride0,
                                                             int outStride1,
                                                             int outStride2,
                                                             int outStride3,
                                                             int outStride4,
                                                             int outStride5,
                                                             void* __restrict__ ws_global)
{
    (void)GridSize;

    void* p_src2dDesc = ws_global;
    void* p_dst1dDesc = static_cast<char*>(ws_global) + 2048;

    const int dstLengths[6] = {
        outLength0, outLength1, outLength2, outLength3, outLength4, outLength5};
    const int dstStrides[6] = {
        outStride0, outStride1, outStride2, outStride3, outStride4, outStride5};

    const auto tupleDstLengths = make_tuple_from_array(dstLengths, Number<dstDims>{});
    const auto tupleDstStrides = make_tuple_from_array(dstStrides, Number<dstDims>{});

    auto dstDesc = make_naive_tensor_descriptor(tupleDstLengths, tupleDstStrides);

    index_t invariantLen;
    const index_t toReduceLen = BlkGroupSize;

    if constexpr(dstDims == 1)
    {
        invariantLen = dstDesc.GetLength(Number<0>{});

        if(get_thread_local_1d_id() == 0)
            *static_cast<decltype(dstDesc)*>(p_dst1dDesc) = dstDesc;
    }
    else
    {
        auto dst1dDesc = transform_tensor_descriptor(
            dstDesc,
            make_tuple(make_merge_transform(tupleDstLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
            make_tuple(Sequence<0>{}));

        invariantLen = dst1dDesc.GetLength(Number<0>{});

        if(get_thread_local_1d_id() == 0)
            *static_cast<decltype(dst1dDesc)*>(p_dst1dDesc) = dst1dDesc;
    }

    auto src2dDesc = make_naive_tensor_descriptor_packed(make_tuple(invariantLen, toReduceLen));

    constexpr auto copySliceLen = BlockSize * GredAccessesPerThreadInBlock;

    if constexpr(src2d_need_padding)
    {
        const auto srcPad =
            ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

        auto src2dDesc_2 =
            transform_tensor_descriptor(src2dDesc,
                                        make_tuple(make_pass_through_transform(invariantLen),
                                                   make_pad_transform(toReduceLen, 0, srcPad)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
        if(get_thread_local_1d_id() == 0)
            *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2;
    }
    else
    {
        if(get_thread_local_1d_id() == 0)
            *static_cast<decltype(src2dDesc)*>(p_src2dDesc) = src2dDesc;
    }
};

template <index_t dstDims>
struct get_ref_desc_types
{
    using refType_dst1dDesc = typename get_ref_1d_desc_types<dstDims>::refType_1dDesc;

    static constexpr index_t ref_invariantLen = refType_dst1dDesc{}.GetLength(Number<0>{});
    static constexpr index_t ref_toReduceLen  = 8;

    static constexpr auto ref_src2dDesc =
        make_naive_tensor_descriptor_packed(make_tuple(ref_invariantLen, ref_toReduceLen));

    using refType_src2dDesc = decltype(ref_src2dDesc);

    // used by the BlockWise and MultiBlock method
    using refType_src2dDesc_padded_34 = decltype(
        transform_tensor_descriptor(ref_src2dDesc,
                                    make_tuple(make_pass_through_transform(ref_invariantLen),
                                               make_pad_transform(ref_toReduceLen, 0, 2)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{})));

    using refType_dst1dDesc_padded =
        decltype(transform_tensor_descriptor(refType_dst1dDesc{},
                                             make_tuple(make_pad_transform(ref_invariantLen, 0, 2)),
                                             make_tuple(Sequence<0>{}),
                                             make_tuple(Sequence<0>{})));
};

using refType_src2dDesc = typename get_ref_desc_types<dstDims>::refType_src2dDesc;
using refType_dst1dDesc = typename get_ref_desc_types<dstDims>::refType_dst1dDesc;
using refType_src2dDesc_padded_34 =
    typename get_ref_desc_types<dstDims>::refType_src2dDesc_padded_34;
using refType_dst1dDesc_padded = typename get_ref_desc_types<dstDims>::refType_dst1dDesc_padded;

template <bool need_padding>
static __device__ auto get_reduction_src2d_descriptor(const void* p_src2dDesc)
{
    using src2dDescType =
        typename conditional<need_padding, refType_src2dDesc_padded_34, refType_src2dDesc>::type;

    return (*reinterpret_cast<const src2dDescType*>(p_src2dDesc));
};

template <bool need_padding>
static __device__ auto get_reduction_dst1d_descriptor(const void* p_dst1dDesc)
{
    using dst1dDescType =
        typename conditional<need_padding, refType_dst1dDesc_padded, refType_dst1dDesc>::type;

    return (*reinterpret_cast<const dst1dDescType*>(p_dst1dDesc));
};

extern "C" __global__ void gridwise_generic_reduce_2(int origReduceLen,
                                                     float alpha,
                                                     const void* __restrict__ p_src_global,
                                                     float beta,
                                                     void* __restrict__ p_dst_global,
                                                     const void CONSTANT* ws_global,
                                                     long ws_buf2_bytes_offset,
                                                     void* __restrict__ indices_global)
{
    (void)p_src_global;

    const void* p_src2dDesc = cast_pointer_to_generic_address_space(ws_global);
    const void* p_dst1dDesc = static_cast<const char*>(p_src2dDesc) + 2048;
    void* ws_buf1_global    = const_cast<char*>(static_cast<const char*>(p_src2dDesc) + 4096);

    const auto src2dDesc = get_reduction_src2d_descriptor<src2d_need_padding>(p_src2dDesc);
    const auto dst1dDesc = get_reduction_dst1d_descriptor<dst1d_need_padding>(p_dst1dDesc);

    using gridwise_2d_reduce = GridwiseReduction_xy_to_x_blockwise<BlockSize,
                                                                   srcDataType,
                                                                   dstDataType,
                                                                   compType,
                                                                   decltype(src2dDesc),
                                                                   decltype(dst1dDesc),
                                                                   op,
                                                                   nanPropaOpt,
                                                                   reduceIndicesOpt,
                                                                   false,
                                                                   true,
                                                                   GredAccessesPerThreadInBlock>;

    void* const ws_buf2_global =
        ws_buf2_bytes_offset > 0
            ? static_cast<void*>(static_cast<char*>(ws_buf1_global) + ws_buf2_bytes_offset)
            : nullptr;

    constexpr int RunId = need_indices ? 3 : 1;
    gridwise_2d_reduce::template Run<RunId>(
        src2dDesc,
        dst1dDesc,
        origReduceLen,
        alpha,
        static_cast<const srcDataType* const __restrict__>(ws_buf1_global),
        beta,
        static_cast<dstDataType* const __restrict__>(p_dst_global),
        static_cast<const int* const __restrict__>(ws_buf2_global),
        static_cast<int* const __restrict__>(indices_global));
};
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef CK_DYNAMIC_GRIDWISE_GENERIC_2D_REDUCTION_DIRECT_WARPWISE_HPP
#define CK_DYNAMIC_GRIDWISE_GENERIC_2D_REDUCTION_DIRECT_WARPWISE_HPP

#include "float_type.hpp"
#include "dynamic_reduction_operator.hpp"
#include "dynamic_reduction_functions.hpp"
#include "reduction_common.hpp"

#include "threadwise_dynamic_tensor_slice_transfer.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType,
          typename dstDataType,
          typename src2dDescType,
          typename dst1dDescType,
          typename compType,
          ReduceTensorOp_t op,
          NanPropagation_t nanPropaOpt,
          ReduceTensorIndices_t reduceIndicesOpt,
          bool isFirstCall,
          bool isLastCall,
          index_t GredAccessesPerThreadInWarp>
struct GridwiseReduction_xy_to_x_direct_warpwise
{
    static constexpr bool indexable = reduce_binary_operator<compType, op>::indexable;
    static constexpr bool need_indices =
        indexable && (reduceIndicesOpt != ReduceTensorIndices_t::NO_INDICES);

    using opReduce = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOpType = typename reduce_unary_operator<compType, op, isFirstCall, isLastCall>::preUnaryOp;
    using posUnaryOpType = typename reduce_unary_operator<compType, op, isFirstCall, isLastCall>::posUnaryOp;

    __device__ void Run(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, 
		        srcDataType alpha,
                        const srcDataType* const __restrict__ p_src_global,
                        dstDataType beta,
                        dstDataType* const __restrict__ p_dst_global,
                        const int* const __restrict__ ws_indices_global,
                        int* const __restrict__ indices_global)
    {
        static_if<need_indices>{}([&](auto) {
            static_if<isFirstCall>{}([&](auto) {
                RunImpl2(src2dDesc, dst1dDesc, origReduceLen,  alpha, p_src_global, beta, p_dst_global, indices_global);
            }).Else([&](auto) {
                RunImpl3(src2dDesc, dst1dDesc, origReduceLen, alpha, p_src_global, beta, p_dst_global, ws_indices_global, indices_global); 
            });
        }).Else([&](auto) { RunImpl1(src2dDesc, dst1dDesc, origReduceLen, alpha, p_src_global, beta, p_dst_global); });
    };

    __device__ static void RunImpl1(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, 
		                    srcDataType alpha,
                                    const srcDataType* const __restrict__ p_src_global,
                                    dstDataType beta,
                                    dstDataType* const __restrict__ p_dst_global)
    {
        const auto src_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_src_global, src2dDesc.GetElementSpaceSize());
        const auto dst_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_dst_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpace::Vgpr, compType, GredAccessesPerThreadInWarp> in_thread_buf;
        StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;

        auto zeroVal = opReduce::GetZeroVal();

        accuValue_buf(Number<0>{}) = zeroVal;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider = origReduceLen; 

        const preUnaryOpType preUnaryOp(divider);
        const posUnaryOpType posUnaryOp(divider);
	
        using ThreadBufferLengths = Sequence<1, GredAccessesPerThreadInWarp>;
        constexpr auto ThreadBufferDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<1>{}, Number<GredAccessesPerThreadInWarp>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();
        index_t warp_global_1d_id   = thread_global_1d_id / warpSize;
        index_t thread_inwarp_id    = thread_global_1d_id % warpSize;

        auto threadwise_src_load = ThreadwiseDynamicTensorSliceTransfer_v2<
                                                                srcDataType,
                                                                compType,
                                                                src2dDescType,
                                                                decltype(ThreadBufferDesc),
                                                                ThreadBufferLengths,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1,
                                                                false>(src2dDesc, make_multi_index(warp_global_1d_id, thread_inwarp_id * GredAccessesPerThreadInWarp));
	
        using warpwise_reduce = WarpReduce<compType, BlockSize, GredAccessesPerThreadInWarp, opReduce, nanPropaOpt>;

        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += warpSize * GredAccessesPerThreadInWarp)
        {
            // zero the data on the Thread Buffer
            warpwise_reduce::set_buffer_value(in_thread_buf, zeroVal);

            threadwise_src_load.Run(src2dDesc, src_global_buf, ThreadBufferDesc, in_thread_buf, type_convert<srcDataType>{}(zeroVal));	    

            // do element-wise pre-reduction operation
            warpwise_reduce::operate_on_elements(preUnaryOp, in_thread_buf);

            // do the warp-wise reduction on data of all thread buffers
            warpwise_reduce::Reduce(in_thread_buf, accuValue_buf(Number<0>{}));

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, warpSize * GredAccessesPerThreadInWarp>{});
        }

        posUnaryOp(accuValue_buf(Number<0>{}));

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<1>{}));

        // The first thread in the warp stores the reduced result to the global location
        // representing the Warp
        if(thread_inwarp_id == 0)
        {
            if(!float_equal_one{}(alpha))
                accuValue_buf[0] *= type_convert<compType>{}(alpha);

            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load = ThreadwiseDynamicTensorSliceTransfer_v2<
                                                                   dstDataType,
                                                                   dstDataType,
                                                                   dst1dDescType,
                                                                   decltype(ReducedDataDesc),
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));

                StaticBuffer<AddressSpace::Vgpr, dstDataType, 1> priorDstValue_buf;

                threadwise_dst_load.Run(dst1dDesc, dst_global_buf, ReducedDataDesc, priorDstValue_buf, type_convert<dstDataType>{}(zeroVal));

                accuValue_buf(Number<0>{}) += type_convert<compType>{}(priorDstValue_buf(Number<0>{}) * beta);
            }

            auto threadwise_dst_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   compType,
                                                                   dstDataType,
                                                                   decltype(ReducedDataDesc),
                                                                   dst1dDescType,
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));


            threadwise_dst_store.Run(ReducedDataDesc, accuValue_buf, dst1dDesc, dst_global_buf, zeroVal);
        }
    };

    __device__ static void RunImpl2(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, 
		                    srcDataType alpha,
                                    const srcDataType* const __restrict__ p_src_global,
                                    dstDataType beta,
                                    dstDataType* const __restrict__ p_dst_global,
                                    int* const __restrict__ indices_global)
    {
        const auto src_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_src_global, src2dDesc.GetElementSpaceSize());
        const auto dst_global_val_buf = make_dynamic_buffer<AddressSpace::Global>(p_dst_global, dst1dDesc.GetElementSpaceSize());
        const auto dst_global_idx_buf = make_dynamic_buffer<AddressSpace::Global>(indices_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpace::Vgpr, compType, GredAccessesPerThreadInWarp> in_thread_buf;
        StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;
        StaticBuffer<AddressSpace::Vgpr, int, 1> accuIndex_buf;

        auto zeroVal       = opReduce::GetZeroVal();

        accuValue_buf(Number<0>{}) = zeroVal;
        accuIndex_buf(Number<0>{}) = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        using ThreadBufferLengths = Sequence<1, GredAccessesPerThreadInWarp>;
        constexpr auto ThreadBufferDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<1>{}, Number<GredAccessesPerThreadInWarp>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();
        index_t warp_global_1d_id   = thread_global_1d_id / warpSize;
        index_t thread_inwarp_id    = thread_global_1d_id % warpSize;

        auto threadwise_src_load = ThreadwiseDynamicTensorSliceTransfer_v2<
                                                                srcDataType,
                                                                compType,
                                                                src2dDescType,
                                                                decltype(ThreadBufferDesc),
                                                                ThreadBufferLengths,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1,
                                                                false>(src2dDesc, make_multi_index(warp_global_1d_id, thread_inwarp_id * GredAccessesPerThreadInWarp));
	
        using warpwise_reduce = WarpReduce<compType, BlockSize, GredAccessesPerThreadInWarp, opReduce, nanPropaOpt>;

        index_t indexOffset = 0;
        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += warpSize * GredAccessesPerThreadInWarp)
        {
            // zero the data on the Thread Buffer
            warpwise_reduce::set_buffer_value(in_thread_buf, zeroVal);

            threadwise_src_load.Run(src2dDesc, src_global_buf, in_thread_buf, type_convert<srcDataType>{}(zeroVal));

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            warpwise_reduce::operate_on_elements(preUnaryOp, in_thread_buf);

            // do the warp-wise reduction on data of all thread buffers
            warpwise_reduce::Reduce2(in_thread_buf, accuValue_buf(Number<0>{}), accuIndex_buf(Number<0>{}), indexOffset);

            indexOffset += warpSize * GredAccessesPerThreadInWarp;

            threadwise_src_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, warpSize * GredAccessesPerThreadInWarp>{});
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<1>{}));

        // The first thread in the warp stores the reduced result to the global location
        // representing the Warp
        if(thread_inwarp_id == 0)
        {
            if(!float_equal_one{}(alpha))
                accuValue_buf(Number<0>{}) *= type_convert<compType>{}(alpha);

            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load = ThreadwiseDynamicTensorSliceTransfer_v2<
                                                                   dstDataType,
                                                                   dstDataType,
                                                                   dst1dDescType,
                                                                   decltype(ReducedDataDesc),
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));

                StaticBuffer<AddressSpace::Vgpr, dstDataType, 1> priorDstValue_buf;

                threadwise_dst_load.Run(dst1dDesc, dst_global_val_buf, ReducedDataDesc, priorDstValue_buf, type_convert<dstDataType>{}(zeroVal));

                accuValue_buf(Number<0>{}) += type_convert<compType>{}(priorDstValue_buf[Number<0>{}] * beta);
            }

            auto threadwise_dst_val_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   compType,
                                                                   dstDataType,
                                                                   decltype(ReducedDataDesc),
                                                                   dst1dDescType,
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));

            auto threadwise_dst_idx_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   int,
                                                                   int,
                                                                   decltype(ReducedDataDesc),
                                                                   dst1dDescType,
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));

            threadwise_dst_val_store.Run(ReducedDataDesc, accuValue_buf, dst1dDesc, dst_global_val_buf, zeroVal);
            threadwise_dst_idx_store.Run(ReducedDataDesc, accuIndex_buf, dst1dDesc, dst_global_idx_buf, 0);
        }
    };

    __device__ static void RunImpl3(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, 
		                    srcDataType alpha,
                                    const srcDataType* const __restrict__ ws_values_global,
                                    dstDataType beta,
                                    dstDataType* const __restrict__ p_dst_global,
                                    const int* const __restrict__ ws_indices_global,
                                    int* const __restrict__ indices_global)
    {
        (void)origReduceLen; 

        const auto src_global_val_buf = make_dynamic_buffer<AddressSpace::Global>(ws_values_global, src2dDesc.GetElementSpaceSize());
        const auto src_global_idx_buf = make_dynamic_buffer<AddressSpace::Global>(ws_indices_global, src2dDesc.GetElementSpaceSize());
        const auto dst_global_val_buf = make_dynamic_buffer<AddressSpace::Global>(p_dst_global, dst1dDesc.GetElementSpaceSize());
        const auto dst_global_idx_buf = make_dynamic_buffer<AddressSpace::Global>(indices_global, dst1dDesc.GetElementSpaceSize());

        StaticBuffer<AddressSpace::Vgpr, compType, GredAccessesPerThreadInWarp> in_thread_val_buf;
        StaticBuffer<AddressSpace::Vgpr, int, GredAccessesPerThreadInWarp> in_thread_idx_buf;
        StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;
        StaticBuffer<AddressSpace::Vgpr, int, 1> accuIndex_buf;
	
        auto zeroVal       = opReduce::GetZeroVal();

        accuValue_buf(Number<0>{}) = zeroVal;
        accuIndex_buf(Number<0>{}) = 0;
	
        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});

        using ThreadBufferLengths = Sequence<1, GredAccessesPerThreadInWarp>;
        constexpr auto ThreadBufferDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<1>{}, Number<GredAccessesPerThreadInWarp>{}));

        index_t thread_global_1d_id = get_block_1d_id() * BlockSize + get_thread_local_1d_id();
        index_t warp_global_1d_id   = thread_global_1d_id / warpSize;
        index_t thread_inwarp_id    = thread_global_1d_id % warpSize;

        auto threadwise_src_val_load = ThreadwiseDynamicTensorSliceTransfer_v2<
                                                                srcDataType,
                                                                compType,
                                                                src2dDescType,
                                                                decltype(ThreadBufferDesc),
                                                                ThreadBufferLengths,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1,
                                                                false>(src2dDesc, make_multi_index(warp_global_1d_id, thread_inwarp_id * GredAccessesPerThreadInWarp));

        auto threadwise_src_idx_load = ThreadwiseDynamicTensorSliceTransfer_v2<
                                                                int,
                                                                int,
                                                                src2dDescType,
                                                                decltype(ThreadBufferDesc),
                                                                ThreadBufferLengths,
                                                                Sequence<0, 1>,
                                                                1,
                                                                1,
                                                                1,
                                                                false>(src2dDesc, make_multi_index(warp_global_1d_id, thread_inwarp_id * GredAccessesPerThreadInWarp));

        using warpwise_reduce = WarpReduce<compType, BlockSize, GredAccessesPerThreadInWarp, opReduce, nanPropaOpt>;

        // zero the data on the Thread Buffer
        warpwise_reduce::set_buffer_value(in_thread_val_buf, zeroVal);

        for(index_t reducedLength = 0; reducedLength < toReduceLength;
            reducedLength += warpSize * GredAccessesPerThreadInWarp)
        {
            threadwise_src_val_load.Run(src2dDesc, src_global_val_buf, in_thread_val_buf, type_convert<srcDataType>{}(zeroVal));
            threadwise_src_idx_load.Run(src2dDesc, src_global_idx_buf, in_thread_idx_buf, static_cast<int>(0));

            // do the warp-wise reduction on data of all thread buffers
            warpwise_reduce::Reduce3(in_thread_val_buf, in_thread_idx_buf, accuValue_buf(Number<0>{}), accuIndex_buf(Number<0>{}));

            // zero the data on the Thread Buffer
            warpwise_reduce::set_buffer_value(in_thread_val_buf, zeroVal);

            threadwise_src_val_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, warpSize * GredAccessesPerThreadInWarp>{});
            threadwise_src_idx_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, warpSize * GredAccessesPerThreadInWarp>{});
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(Number<1>{}));

        // The first thread in the warp stores the reduced result to the global location
        // representing the Warp
        if(thread_inwarp_id == 0)
        {
            if(!float_equal_one{}(alpha))
                accuValue_buf(Number<0>{}) *= type_convert<compType>{}(alpha);

            if(!float_equal_zero{}(beta))
            {
                auto threadwise_dst_load = ThreadwiseDynamicTensorSliceTransfer_v2<
                                                                   dstDataType,
                                                                   dstDataType,
                                                                   dst1dDescType,
                                                                   decltype(ReducedDataDesc),
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));

                StaticBuffer<AddressSpace::Vgpr, dstDataType, 1> priorDstValue_buf;

                threadwise_dst_load.Run(dst1dDesc, dst_global_val_buf, ReducedDataDesc, priorDstValue_buf, type_convert<dstDataType>{}(zeroVal));

                accuValue_buf(Number<0>{}) += type_convert<compType>{}(priorDstValue_buf[Number<0>{}] * beta);
            }

            auto threadwise_dst_val_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   compType,
                                                                   dstDataType,
                                                                   decltype(ReducedDataDesc),
                                                                   dst1dDescType,
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));

            auto threadwise_dst_idx_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   int,
                                                                   int,
                                                                   decltype(ReducedDataDesc),
                                                                   dst1dDescType,
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1dDesc, make_multi_index(warp_global_1d_id));

            threadwise_dst_val_store.Run(ReducedDataDesc, accuValue_buf, dst1dDesc, dst_global_val_buf, zeroVal);
            threadwise_dst_idx_store.Run(ReducedDataDesc, accuIndex_buf, dst1dDesc, dst_global_idx_buf, 0);
        }
    };
};

} // namespace ck
#endif

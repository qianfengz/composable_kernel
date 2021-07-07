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
#ifndef CK_DYNAMIC_GRIDWISE_GENERIC_2D_REDUCTION_BLOCKWISE_HPP
#define CK_DYNAMIC_GRIDWISE_GENERIC_2D_REDUCTION_BLOCKWISE_HPP

#include "float_type.hpp"
#include "dynamic_reduction_operator.hpp"
#include "dynamic_reduction_functions.hpp"
#include "reduction_common.hpp"

#include "blockwise_dynamic_generic_tensor_slice_transfer.hpp"
#include "ConstantMatrixDescriptor.hpp"

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
          index_t GredAccessesPerThreadInBlock>
struct GridwiseReduction_xy_to_x_blockwise
{
    static constexpr bool indexable = reduce_binary_operator<compType, op>::indexable;
    static constexpr bool need_indices =
        indexable && (reduceIndicesOpt != ReduceTensorIndices_t::NO_INDICES);

    static constexpr index_t BlockBufferSize = BlockSize * GredAccessesPerThreadInBlock;

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
                RunImpl2(src2dDesc, dst1dDesc, origReduceLen, alpha, p_src_global, beta, p_dst_global, indices_global);
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
        // LDS
        __shared__ compType p_in_block_buffer[BlockBufferSize];
        
        const auto src_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_src_global, src2dDesc.GetElementSpaceSize());
        const auto dst_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_dst_global, dst1dDesc.GetElementSpaceSize());

        const auto in_block_buf = make_dynamic_buffer<AddressSpace::Lds>(p_in_block_buffer, BlockBufferSize);	
	StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;

        auto zeroVal       = opReduce::GetZeroVal();
        accuValue_buf[0] = zeroVal;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);
        const posUnaryOpType posUnaryOp(divider);

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();

        constexpr auto in_block_desc = make_native_tensor_descriptor_packed(Sequence<1, BlockBufferSize>{});

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_load =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<1, BlockBufferSize>,
                                                   ThreadSliceLengths,
                                                   ThreadClusterLengths,
                                                   Sequence<0, 1>,
                                                   srcDataType,
                                                   compType,
                                                   src2dDescType,
                                                   decltype(in_block_desc),
                                                   Sequence<0, 1>,
                                                   Sequence<0, 1>,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   false,
                                                   true>(src2dDesc,
                                                         make_multi_index(block_global_1d_id, 0),
                                                         in_block_desc,
                                                         make_multi_index(0, 0));	

        constexpr auto block_buff_2d_desc = make_native_tensor_descriptor_packed(
            Sequence<GredAccessesPerThreadInBlock, BlockSize>{});

        using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_2d_desc),
                                                                    compType,
                                                                    true,
                                                                    opReduce,
                                                                    nanPropaOpt>;

        const index_t toReduceBlocks = (toReduceLength + BlockSize - 1) / BlockSize;

        for(index_t reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::set_buffer_value(in_block_buf, zeroVal);

            blockwise_src_load.RunRead(src2dDesc, src_global_buf, type_convert<srcDataType>{}(zeroVal));
            blockwise_src_load.RunWrite(in_block_desc, in_block_buf, type_convert<compType>{}(zeroVal)); 

            __syncthreads();

            // do element-wise pre-reduction operation
            blockwise_reduce::operate_on_elements(preUnaryOp, in_block_buff);

            index_t BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                        ? GredAccessesPerThreadInBlock
                                        : toReduceBlocks - reducedBlocks;
            blockwise_reduce::Reduce(in_block_buf, BlocksInOneOp, accuValue_buf[0]);

            blockwise_src_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, BlockBufferSize>{});
        }

        posUnaryOp(accuValue_buf[0]);

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
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
                                                                   false>(dst1Desc, make_multi_index(block_global_1d_id));

                StaticBuffer<AddressSpace::Vgpr, dstDataType, 1> priorDstValue_buf;

                threadwise_dst_load.Run(dst1dDesc, dst_global_buf, ReducedDataDesc, priorDstValue_buf, type_convert<dstDataType>{}(zeroVal));

                accuValue_buffer[0] += type_convert<compType>{}(priorDstValue_buf[0] * beta);
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
                                                                   false>(dst1Desc, make_multi_index(thread_global_1d_id));

            threadwise_dst_store.Run(ReducedDataDesc, &accuValue_buf[0], dst1dDesc, dst_global_buf, zeroVal);
        }
    };

    __device__ static void RunImpl2(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, 
		                    srcDataType alpha,
                                    const srcDataType* const __restrict__ p_src_global,
                                    dstDataType beta,
                                    dstDataType* const __restrict__ p_dst_global,
                                    int* const __restrict__ indices_global)
    {
        // LDS
        __shared__ compType p_in_block_buffer[BlockBufferSize];
        __shared__ int block_indices_buffer[BlockBufferSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_src_global, src2dDesc.GetElementSpaceSize());
        const auto dst_global_val_buf = make_dynamic_buffer<AddressSpace::Global>(p_dst_global, dst1dDesc.GetElementSpaceSize());
        const auto dst_global_idx_buf = make_dynamic_buffer<AddressSpace::Global>(indices_global, dst1dDesc.GetElementSpaceSize());

        const auto in_block_val_buf = make_dynamic_buffer<AddressSpace::Lds>(p_in_block_buffer, BlockBufferSize);
        const auto in_block_idx_buf = make_dynamic_buffer<AddressSpace::Lds>(block_indices_buffer, BlockBufferSize);
	
        StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;
        StaticBuffer<AddressSpace::Vgpr, int, 1> accuIndex_buf;
	
        auto zeroVal       = opReduce::GetZeroVal();

        accuValue_buf[0] = zeroVal;
        accuIndex_buf[0] = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{}); 
        const int divider = origReduceLen; 

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{}); 

        constexpr auto in_block_desc =
            make_native_tensor_descriptor_packed(Sequence<1, BlockBufferSize>{});

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_load =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<1, BlockBufferSize>,
                                                   ThreadSliceLengths,
                                                   ThreadClusterLengths,
                                                   Sequence<0, 1>,
                                                   srcDataType,
                                                   dstDataType,
                                                   src2dDescType,
                                                   decltype(in_block_desc),
                                                   Sequence<0, 1>,
                                                   Sequence<0, 1>,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   false,
                                                   true>(src2dDesc,
                                                         make_multi_index(block_global_1d_id, 0),
                                                         in_block_desc,
                                                         make_multi_index(0, 0));

        constexpr auto block_buff_2d_desc = make_native_tensor_descriptor_packed(
            Sequence<GredAccessesPerThreadInBlock, BlockSize>{});

        using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_2d_desc),
                                                                    compType,
                                                                    true,
                                                                    opReduce,
                                                                    nanPropaOpt>;

        const index_t toReduceBlocks = (toReduceLength + BlockSize - 1) / BlockSize;

        int indexOffset = 0;

        for(index_t reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::set_buffer_value(p_in_block_buffer, zeroVal);

            // load block data from global to LDS, no use of double buffers (to be improved)
            blockwise_src_load.RunRead(src2dDesc, src_global_val_buf, type_convert<srcDataType>{}(zeroVal));
            blockwise_src_load.RunWrite(in_block_desc, in_block_val_buf, zeroVal);

            __syncthreads();

            // construct the indices for the current toReduce blocks
            blockwise_reduce::init_buffer_indices(in_block_idx_buf, indexOffset);

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            blockwise_reduce::operate_on_elements(preUnaryOp, in_block_val_buf);

            index_t BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                        ? GredAccessesPerThreadInBlock
                                        : toReduceBlocks - reducedBlocks;

            blockwise_reduce::Reduce2(in_block_val_buf, in_block_idx_buf, BlocksInOneOp, accuValue_buf[0], accuIndex_buf[0]);

            indexOffset += BlockBufferSize;

            blockwise_src_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, BlockBufferSize>{});
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            if(!float_equal_one{}(alpha))
                accuValue *= type_convert<compType>{}(alpha);

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
                                                                   false>(dst1Desc, make_multi_index(block_global_1d_id));

                StaticBuffer<AddressSpace::Vgpr, dstDataType, 1> priorDstValue_buf;

                threadwise_dst_load.Run(dst1dDesc, dst_global_buf, ReducedDataDesc, priorDstValue_buf, type_convert<dstDataType>{}(zeroVal));

                accuValue_buffer[0] += type_convert<compType>{}(priorDstValue_buf[0] * beta);
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
                                                                   false>(dst1Desc, make_multi_index(thread_global_1d_id));

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
                                                                   false>(dst1Desc, make_multi_index(thread_global_1d_id));

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

        // LDS
        __shared__ compType p_in_block_buffer[BlockBufferSize];
        __shared__ int block_indices_buffer[BlockBufferSize];

        const auto src_global_val_buf = make_dynamic_buffer<AddressSpace::Global>(ws_values_global, src2dDesc.GetElementSpaceSize());
        const auto src_global_idx_buf = make_dynamic_buffer<AddressSpace::Global>(ws_indices_global, src2dDesc.GetElementSpaceSize());	
        const auto dst_global_val_buf = make_dynamic_buffer<AddressSpace::Global>(p_dst_global, dst1dDesc.GetElementSpaceSize());
        const auto dst_global_idx_buf = make_dynamic_buffer<AddressSpace::Global>(indices_global, dst1dDesc.GetElementSpaceSize());

        const auto in_block_val_buf = make_dynamic_buffer<AddressSpace::Lds>(p_in_block_buffer, BlockBufferSize);
        const auto in_block_idx_buf = make_dynamic_buffer<AddressSpace::Lds>(block_indices_buffer, BlockBufferSize);

        StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;
        StaticBuffer<AddressSpace::Vgpr, int, 1> accuIndex_buf;

        auto zeroVal       = opReduce::GetZeroVal();

        accuValue_buf[0] = zeroVal;
        accuIndex_buf[0] = 0;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});	

        const index_t thread_local_id    = get_thread_local_1d_id();
        const index_t block_global_1d_id = get_block_1d_id();

        constexpr auto in_block_desc = make_native_tensor_descriptor_packed(Sequence<1, BlockBufferSize>{});

        using ThreadSliceLengths   = Sequence<1, GredAccessesPerThreadInBlock>;
        using ThreadClusterLengths = Sequence<1, BlockSize>;

        auto blockwise_src_val_load =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<1, BlockBufferSize>,
                                                   ThreadSliceLengths,
                                                   ThreadClusterLengths,
                                                   Sequence<0, 1>,
                                                   srcDataType,
                                                   compType,
                                                   src2dDescType,
                                                   decltype(in_block_desc),
                                                   Sequence<0, 1>,
                                                   Sequence<0, 1>,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   false,
                                                   true>(src2dDesc,
                                                         make_multi_index(block_global_1d_id, 0),
                                                         in_block_desc,
                                                         make_multi_index(0, 0));

        auto blockwise_src_idx_load =
            BlockwiseDynamicTensorSliceTransfer_v4<BlockSize,
                                                   InMemoryDataOperation::Set,
                                                   Sequence<1, BlockBufferSize>,
                                                   ThreadSliceLengths,
                                                   ThreadClusterLengths,
                                                   Sequence<0, 1>,
                                                   int,
                                                   int,
                                                   src2dDescType,
                                                   decltype(in_block_desc),
                                                   Sequence<0, 1>,
                                                   Sequence<0, 1>,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   1,
                                                   false,
                                                   true>(src2dDesc,
                                                         make_multi_index(block_global_1d_id, 0),
                                                         in_block_desc,
                                                         make_multi_index(0, 0));

        constexpr auto block_buff_2d_desc = make_native_tensor_descriptor_packed(
            Sequence<GredAccessesPerThreadInBlock, BlockSize>{});

        using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_2d_desc),
                                                                    compType,
                                                                    true,
                                                                    opReduce,
                                                                    nanPropaOpt>;

        const index_t toReduceBlocks = (toReduceLength + BlockSize - 1) / BlockSize;

        for(index_t reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::set_buffer_value(in_block_val_buf, zeroVal);

            // load block data from global to LDS, no use of double buffers (to be improved)
            blockwise_src_val_load.RunRead(src2dDesc, src_global_val_buf, type_convert<srcDataType>{}(zeroVal));
            blockwise_src_idx_load.RunRead(src2dDesc, src_global_idx_buf, static_cast<int>(0));
            blockwise_src_val_load.RunWrite(in_block_desc, in_block_val_buf, zeroVal);
            blockwise_src_idx_load.RunWrite(in_block_desc, in_block_idx_buf, static_cast<int>(0));

            __syncthreads();

            index_t BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                        ? GredAccessesPerThreadInBlock
                                        : toReduceBlocks - reducedBlocks;

            blockwise_reduce::Reduce2(in_block_val_buf, in_block_idx_buf, BlocksInOneOp, accuValue, accuIndex);

            blockwise_src_val_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, BlockSize * GredAccessesPerThreadInBlock>{});
            blockwise_src_idx_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, BlockSize * GredAccessesPerThreadInBlock>{});
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_native_tensor_descriptor_packed(ReducedDataLengths{});

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
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
                                                                   false>(dst1Desc, make_multi_index(block_global_1d_id));

                StaticBuffer<AddressSpace::Vgpr, dstDataType, 1> priorDstValue_buf;

                threadwise_dst_load.Run(dst1dDesc, dst_global_val_buf, ReducedDataDesc, priorDstValue_buf, type_convert<dstDataType>{}(zeroVal));

                accuValue_buffer[0] += type_convert<compType>{}(priorDstValue_buf[0] * beta);
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
                                                                   false>(dst1Desc, make_multi_index(block_global_1d_id));

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
                                                                   false>(dst1Desc, make_multi_index(block_global_1d_id));

            threadwise_dst_val_store.Run(ReducedDataDesc, accuValue_buf, dst1dDesc, dst_global_val_buf, zeroVal);
            threadwise_dst_idx_store.Run(ReducedDataDesc, accuIndex_buf, dst1dDesc, dst_global_idx_buf, 0);
        }
    };
};

} // namespace ck
#endif
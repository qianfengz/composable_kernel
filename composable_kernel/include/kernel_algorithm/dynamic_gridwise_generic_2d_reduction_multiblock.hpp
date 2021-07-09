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
#ifndef CK_DYNAMIC_GRIDWISE_GENERIC_2D_REDUCTION_MULTIBLOCK_HPP
#define CK_DYNAMIC_GRIDWISE_GENERIC_2D_REDUCTION_MULTIBLOCK_HPP

#include "float_type.hpp"
#include "dynamic_reduction_operator.hpp"
#include "dynamic_reduction_functions.hpp"
#include "reduction_common.hpp"

#include "blockwise_dynamic_tensor_slice_transfer.hpp"
#include "ConstantMatrixDescriptor.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType,
          typename dstDataType, // not used together with the beta input
          typename src2dDescType,
          typename dst1dDescType,
          typename compType,
          ReduceTensorOp_t op,
          NanPropagation_t nanPropaOpt,
          ReduceTensorIndices_t reduceIndicesOpt,
          index_t GredAccessesPerThreadInBlock>
struct GridwiseReduction_xy_to_x_multiblock
{
    static constexpr bool indexable = reduce_binary_operator<compType, op>::indexable;
    static constexpr bool need_indices =
        indexable && (reduceIndicesOpt != ReduceTensorIndices_t::NO_INDICES);

    static constexpr auto toReduceLength = src2dDesc::GetLength(Number<1>{});

    using opReduce = typename reduce_binary_operator<compType, op>::opType;
    using preUnaryOpType = typename reduce_unary_operator<compType, op, true, false>::preUnaryOp;
    using posUnaryOpType = typename reduce_unary_operator<compType, op, true, false>::posUnaryOp;

    __device__ void Run(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, int BlkGroupSize,
		        srcDataType alpha,
                        const srcDataType* const __restrict__ p_src_global,
                        dstDataType beta,
                        srcDataType* const __restrict__ workspace_global,
                        int* const __restrict__ ws_indices_global)
    {
        static_if<need_indices>{}([&](auto) {
            RunImpl2(src2dDesc, dst1dDesc, origReduceLen, BlkGroupSize, alpha, p_src_global, beta, workspace_global, ws_indices_global);
        }).Else([&](auto) { RunImpl1(src2dDesc, dst1dDesc, origReduceLen, BlkGroupSize, alpha, p_src_global, beta, workspace_global); });
    };

    __device__ static void RunImpl1(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, int BlkGroupSize,
		                    srcDataType alpha,
                                    const srcDataType* const __restrict__ p_src_global,
                                    dstDataType beta,
                                    srcDataType* const __restrict__ workspace_global)
    {
        (void)alpha; // unused
        (void)beta;  // unused

        constexpr index_t BlockBufferSize = BlockSize * GredAccessesPerThreadInBlock;

        // LDS
        __shared__ compType p_in_block_buffer[BlockBufferSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_src_global, src2dDesc.GetElementSpaceSize());
        const auto workspace_global_buf = make_dynamic_buffer<AddressSpace::Global>(workspace_global, dst1dDesc.GetLength(Number<0>{}) * BlkGroupSize);

        const auto in_block_buf = make_dynamic_buffer<AddressSpace::Lds>(p_in_block_buffer, BlockBufferSize);
        StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;

        auto zeroVal = opReduce::GetZeroVal();
        accuValue_buf[0] = zeroVal;

        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);

        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;

        constexpr index_t reduceSizePerBlock =
            (((toReduceLength + BlkGroupSize - 1) / BlkGroupSize + BlockBufferSize - 1) / BlockBufferSize) * BlockBufferSize;

        constexpr auto in_block_desc = make_dynamic_naive_tensor_descriptor_packed(make_tuple(1, BlockSize * GredAccessesPerThreadInBlock));

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
                                                         make_multi_index(blkgroup_id, block_local_id * reduceSizePerBlock),
                                                         in_block_desc,
                                                         make_multi_index(0, 0));

        constexpr auto block_buff_2d_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(GredAccessesPerThreadInBlock, BlockSize));

        using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_2d_desc),
                                                                    compType,
                                                                    true,
                                                                    opReduce,
                                                                    nanPropaOpt>;

        constexpr index_t toReduceBlocks = (reduceSizePerBlock + BlockSize - 1) / BlockSize;

        for(index_t reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::set_buffer_value(in_block_buf, zeroVal);

            blockwise_src_load.RunRead(src2dDesc, src_global_buf, type_convert<srcDataType>{}(zeroVal));
            blockwise_src_load.RunWrite(in_block_desc, in_block_buf, zeroVal);
            __syncthreads();

            // do element-wise pre-reduction operation
            blockwise_reduce::operate_on_elements(preUnaryOp, in_block_buf);

            index_t BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                        ? GredAccessesPerThreadInBlock
                                        : toReduceBlocks - reducedBlocks;
            blockwise_reduce::Reduce(in_block_buf, BlocksInOneOp, accuValue);

            blockwise_src_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, BlockBufferSize>{});
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_naive_tensor_descriptor_packed(ReducedDataLengths{});

        const auto workspace_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(dst1dDesc.GetLength(Number<0>{}) * BlkGroupSize));

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   compType,
                                                                   compType,
                                                                   decltype(ReducedDataDesc),
                                                                   decltype(workspace_desc),
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1Desc, make_multi_index(block_global_id));

            threadwise_workspace_store.Run(ReducedDataDesc, accuValue_buf, workspace_desc, workspace_global_buf, zeroVal);	    
        }
    };

    __device__ static void RunImpl2(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, int origReduceLen, int BlkGroupSize,
		                    srcDataType alpha,
                                    const srcDataType* const __restrict__ p_src_global,
                                    dstDataType beta,
                                    srcDataType* const __restrict__ ws_values_global,
                                    int* const __restrict__ ws_indices_global)
    {
        (void)alpha; // unused
        (void)beta;  // unused

        constexpr index_t BlockBufferSize = BlockSize * GredAccessesPerThreadInBlock;

        // LDS
        __shared__ compType p_in_block_values_buffer[BlockBufferSize];
        __shared__ int p_in_block_indices_buffer[BlockBufferSize];

        const auto src_global_buf = make_dynamic_buffer<AddressSpace::Global>(p_src_global, src2dDesc.GetElementSpaceSize());
        const auto workspace_global_val_buf = make_dynamic_buffer<AddressSpace::Global>(ws_values_global, dst1dDesc.GetLength(Number<0>{}) * BlkGroupSize);
        const auto workspace_global_idx_buf = make_dynamic_buffer<AddressSpace::Global>(ws_indices_global, dst1dDesc.GetLength(Number<0>{}) * BlkGroupSize);

        const auto in_block_val_buf = make_dynamic_buffer<AddressSpace::Lds>(p_in_block_values_buffer, BlockBufferSize);
        const auto in_block_idx_buf = make_dynamic_buffer<AddressSpace::Lds>(p_in_block_indices_buffer, BlockBufferSize);
        StaticBuffer<AddressSpace::Vgpr, compType, 1> accuValue_buf;
        StaticBuffer<AddressSpace::Vgpr, int, 1> accuIndex_buf;

        auto zeroVal = opReduce::GetZeroVal();
        accuValue_buf[0] = zeroVal;
        accuIndex_buf[0] = 0; 
	
        const auto toReduceLength = src2dDesc.GetLength(Number<1>{});
        const int divider = origReduceLen;

        const preUnaryOpType preUnaryOp(divider);
	
        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();
        const index_t blkgroup_id     = block_global_id / BlkGroupSize;
        const index_t block_local_id  = block_global_id % BlkGroupSize;

        constexpr index_t reduceSizePerBlock =
            (((toReduceLength + BlkGroupSize - 1) / BlkGroupSize + BlockBufferSize - 1) / BlockBufferSize) * BlockBufferSize;

        constexpr auto in_block_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(1, BlockSize * GredAccessesPerThreadInBlock));

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
                                                         make_multi_index(blkgroup_id, block_local_id * reduceSizePerBlock),
                                                         in_block_desc,
                                                         make_multi_index(0, 0));

        constexpr auto block_buff_2d_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(GredAccessesPerThreadInBlock, BlockSize));

        using blockwise_reduce = BlockwiseReduction_2d_block_buffer<decltype(block_buff_2d_desc),
                                                                    compType,
                                                                    true,
                                                                    opReduce,
                                                                    nanPropaOpt>;

        constexpr index_t toReduceBlocks = (reduceSizePerBlock + BlockSize - 1) / BlockSize;

        blockwise_reduce::set_buffer_value(in_block_val_buf, zeroVal);

        int indexOffset = block_local_id * reduceSizePerBlock;

        for(index_t reducedBlocks = 0; reducedBlocks < toReduceBlocks;
            reducedBlocks += GredAccessesPerThreadInBlock)
        {
            blockwise_reduce::init_buffer_indices(in_block_idx_buf, indexOffset);

            blockwise_src_load.RunRead(src2dDesc, src_global_buf, in_block_desc, type_convert<srcDataType>{}(zeroVal));
            blockwise_src_load.RunWrite(in_block_desc, in_block_val_buf, zeroVal);

            __syncthreads();

            // unary operation before reducing, needed by AMAX; For MIN/MAX, nothing is actually
            // done here
            blockwise_reduce::operate_on_elements(preUnaryOp, in_block_val_buf);

            index_t BlocksInOneOp = (reducedBlocks < toReduceBlocks - GredAccessesPerThreadInBlock)
                                        ? GredAccessesPerThreadInBlock
                                        : toReduceBlocks - reducedBlocks;

            blockwise_reduce::Reduce2(in_block_val_buf, in_block_idx_buffer, BlocksInOneOp, accuValue_buf[0], accuIndex_buf[0]);

            blockwise_reduce::set_buffer_value(in_block_val_buf, zeroVal);

            indexOffset += BlockBufferSize;

            blockwise_src_load.MoveSrcSliceWindow(src2dDesc, Sequence<0, BlockBufferSize>{});
        }

        using ReducedDataLengths       = Sequence<1>;
        constexpr auto ReducedDataDesc = make_naive_tensor_descriptor_packed(ReducedDataLengths{});

        const auto workspace_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(dst1dDesc.GetLength(Number<0>{}) * BlkGroupSize));

        // The first thread in the block stores the reduced result to the global location
        // representing the block
        if(thread_local_id == 0)
        {
            auto threadwise_workspace_val_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   compType,
                                                                   compType,
                                                                   decltype(ReducedDataDesc),
                                                                   decltype(workspace_desc),
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1Desc, make_multi_index(block_global_id));

            auto threadwise_workspace_idx_store = ThreadwiseDynamicTensorSliceTransfer_v1r3<
                                                                   int,
                                                                   int,
                                                                   decltype(ReducedDataDesc),
                                                                   decltype(workspace_desc),
                                                                   ReducedDataLengths,
                                                                   Sequence<0>,
                                                                   0,
                                                                   1,
                                                                   InMemoryDataOperation::Set,
                                                                   1,
                                                                   false>(dst1Desc, make_multi_index(block_global_id));


            threadwise_workspace_val_store.Run(ReducedDataDesc, accuValue_buf, workspace_desc, workspace_global_val_buf, zeroVal);
            threadwise_workspace_idx_store.Run(ReducedDataDesc, accuIndex_buf, workspace_desc, workspace_global_idx_buf, 0);
        }
    };
};

} // namespace ck
#endif

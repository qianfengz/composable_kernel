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
#ifndef CK_DYNAMIC_GRIDWISE_GENERIC_REDUCTION_HPP
#define CK_DYNAMIC_GRIDWISE_GENERIC_REDUCTION_HPP

#include "float_type.hpp"

#include "dynamic_gridwise_generic_2d_reduction_direct_threadwise.hpp"
#include "dynamic_gridwise_generic_2d_reduction_direct_warpwise.hpp"
#include "dynamic_gridwise_generic_2d_reduction_blockwise.hpp"
#include "dynamic_gridwise_generic_2d_reduction_multiblock.hpp"

namespace ck {

template <index_t BlockSize,
          typename srcDataType,  // the type with which the data of the source tensor are stored
          typename dstDataType,  // the type with which the data of the destintion tensor are stored
          typename compType,     // the type used by the reduce binary operator
          index_t op_I,     // the enumerate value representing the operation used in Reduction
          index_t nanPropaOpt_I,      // the enumerate value representing the NanPropagation Option
          index_t reduceIndicesOpt_I, // the enumerate value representing the Reduce Indices Option
          index_t GredThreadBufferLength,
          index_t GredAccessesPerThreadInBlock,
          index_t GredAccessesPerThreadInWarp>
struct Gridwise2dReduction
{
    static constexpr auto op               = static_cast<ReduceTensorOp_t>(op_I);
    static constexpr auto nanPropaOpt      = static_cast<NanPropagation_t>(nanPropaOpt_I);
    static constexpr auto reduceIndicesOpt = static_cast<ReduceTensorIndices_t>(reduceIndicesOpt_I);

    __device__ Gridwise2dReduction(int reduceImpl_, int origReduceLen_, int BlkGroupSize_)
    {
	reduceImpl = static_cast<ReductionMethod_t>(reduceImpl_); 
	origReduceLen = origReduceLen_; 
	BlkGroupSize = BlkGroupSize_;
    }; 

    // wrapper for switching to the Reduce_DirectThreadWise method
    template <bool isFirstCall, bool isLastCall, typename src2dDescType, typename dst1dDescType>
    __device__ void Run_DirectThreadWise(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc,
                                                srcDataType alpha,
                                                const srcDataType* const __restrict__ p_src_global,
                                                dstDataType beta,
                                                dstDataType* const __restrict__ p_dst_global,
                                                srcDataType* const __restrict__ ws_buf1_global,
                                                int* const __restrict__ ws_buf2_global,
                                                int* const __restrict__ indices_global) const
    {
            (void)ws_buf1_global; // unused

            using gridwise_reduce = GridwiseReduction_xy_to_x_direct_threadwise<BlockSize,
                                                                                srcDataType,
                                                                                dstDataType,
                                                                                src2dDescType,
                                                                                dst1dDescType,
                                                                                compType,
                                                                                op,
                                                                                nanPropaOpt,
                                                                                reduceIndicesOpt,
                                                                                isFirstCall,
                                                                                isLastCall,
                                                                                GredThreadBufferLength>; 
            gridwise_reduce{}.Run(src2dDesc, dst1dDesc, this->origReduceLen, 
			          alpha,
                                  p_src_global,
                                  beta,
                                  p_dst_global,
                                  const_cast<const int* const __restrict__>(ws_buf2_global),
                                  indices_global); // ws_buf2_global will be read at the second-time
    };

    // wrapper for switching to the Reduce_DirectWarpdWise method
    template <bool isFirstCall, bool isLastCall, typename src2dDescType, typename dst1dDescType>
     __device__  void Run_DirectWarpWise(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc,
                                               srcDataType alpha,
                                               const srcDataType* const __restrict__ p_src_global,
                                               dstDataType beta,
                                               dstDataType* const __restrict__ p_dst_global,
                                               srcDataType* const __restrict__ ws_buf1_global,
                                               int* const __restrict__ ws_buf2_global,
                                               int* const __restrict__ indices_global) const 
    {
            (void)ws_buf1_global; // unused

            using gridwise_reduce = GridwiseReduction_xy_to_x_direct_warpwise<BlockSize,
                                                                              srcDataType,
                                                                              dstDataType,
                                                                              src2dDescType,
                                                                              dst1dDescType,
                                                                              compType,
                                                                              op,
                                                                              nanPropaOpt,
                                                                              reduceIndicesOpt,
                                                                              isFirstCall,
                                                                              isLastCall,
                                                                              GredAccessesPerThreadInWarp>; 
            gridwise_reduce{}.Run(src2dDesc, dst1dDesc, this->origReduceLen, 
			          alpha,
                                  p_src_global,
                                  beta,
                                  p_dst_global,
                                  const_cast<const int* const __restrict__>(ws_buf2_global),
                                  indices_global); // ws_buf2_global will be read at the second-time
    };

    // wrapper for switching to the Reduce_BlockWise method
    template <bool isFirstCall, bool isLastCall, typename src2dDescType, typename dst1dDescType>
    __device__  void Run_BlockWise(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc,
                                         srcDataType alpha,
                                         const srcDataType* const __restrict__ p_src_global,
                                         dstDataType beta,
                                         dstDataType* const __restrict__ p_dst_global,
                                         srcDataType* const __restrict__ ws_buf1_global,
                                         int* const __restrict__ ws_buf2_global,
                                         int* const __restrict__ indices_global) const 
    {
            (void)ws_buf1_global; // unused

            using gridwise_reduce = GridwiseReduction_xy_to_x_blockwise<BlockSize,
                                                                        srcDataType,
                                                                        dstDataType,
                                                                        src2dDescType,
                                                                        dst1dDescType,
                                                                        compType,
                                                                        op,
                                                                        nanPropaOpt,
                                                                        reduceIndicesOpt,
                                                                        isFirstCall,
                                                                        isLastCall,
                                                                        GredAccessesPerThreadInBlock>; 
            gridwise_reduce{}.Run(src2dDesc, dst1dDesc, this->origReduceLen,
			          alpha,
                                  p_src_global,
                                  beta,
                                  p_dst_global,
                                  const_cast<const int* const __restrict__>(ws_buf2_global),
                                  indices_global); // ws_buf2_global will be read at the second-time
    };

    // wrapper for switching to the Reduce_MultiBlock method
    template <bool isFirstCall, bool isLastCall, typename src2dDescType, typename dst1dDescType>
    __device__  void Run_MultiBlock(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc,
                                          srcDataType alpha,
                                          const srcDataType* const __restrict__ p_src_global,
                                          dstDataType beta,
                                          dstDataType* const __restrict__ p_dst_global,
                                          srcDataType* const __restrict__ ws_buf1_global,
                                          int* const __restrict__ ws_buf2_global,
                                          int* const __restrict__ indices_global) const 
    {
            (void)p_dst_global;   // unused
            (void)indices_global; // unused

            using gridwise_reduce = GridwiseReduction_xy_to_x_multiblock<BlockSize,
                                                                         srcDataType,
                                                                         dstDataType,
                                                                         src2dDescType,
                                                                         dst1dDescType,
                                                                         compType,
                                                                         op,
                                                                         nanPropaOpt,
                                                                         reduceIndicesOpt,
                                                                         GredAccessesPerThreadInBlock>; 

            gridwise_reduce{}.Run(src2dDesc, dst1dDesc, this->origReduceLen, this->BlkGroupSize,
			          alpha,
                                  p_src_global,
                                  beta,
                                  ws_buf1_global,
                                  ws_buf2_global); // ws_buf1_global instead of p_dst_global,
                                                   // ws_buf2_global instead of indices_global
    };

    template <typename src2dDescType, typename dst1dDescType>
    __device__ void Run(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, 
		               float alpha,
                               const void* const __restrict__ p_src_global,
                               float beta,
                               void* const __restrict__ p_dst_global,
                               void* const __restrict__ ws_buf1_global,
                               long ws_buf2_bytes_offset,
                               void* const __restrict__ indices_global) const
    {
        void* const ws_buf2_global = ws_buf2_bytes_offset > 0 ? static_cast<void*>(static_cast<char*>(ws_buf1_global) + ws_buf2_bytes_offset) : nullptr;

        switch (this->reduceImpl) {
	    case ReductionMethod_t::DirectThreadWise:
                 this->Run_DirectThreadWise<true, true, src2dDescType, dst1dDescType>(src2dDesc, dst1dDesc,
                           type_convert<srcDataType>{}(alpha),
                           static_cast<const srcDataType* const __restrict__>(p_src_global),
                           type_convert<dstDataType>{}(beta),
                           static_cast<dstDataType* const __restrict__>(p_dst_global),
                           static_cast<dstDataType* const __restrict__>(ws_buf1_global),
                           static_cast<int* const __restrict__>(ws_buf2_global),
                           static_cast<int* const __restrict__>(indices_global));
		 break; 
	    case ReductionMethod_t::DirectWarpWise:
                 this->Run_DirectWarpWise<true, true, src2dDescType, dst1dDescType>(src2dDesc, dst1dDesc,
                           type_convert<srcDataType>{}(alpha),
                           static_cast<const srcDataType* const __restrict__>(p_src_global),
                           type_convert<dstDataType>{}(beta),
                           static_cast<dstDataType* const __restrict__>(p_dst_global),
                           static_cast<dstDataType* const __restrict__>(ws_buf1_global),
                           static_cast<int* const __restrict__>(ws_buf2_global),
                           static_cast<int* const __restrict__>(indices_global));
		 break; 
            case ReductionMethod_t::BlockWise:
                 this->Run_BlockWise<true, true, src2dDescType, dst1dDescType>(src2dDesc, dst1dDesc,
                           type_convert<srcDataType>{}(alpha),
                           static_cast<const srcDataType* const __restrict__>(p_src_global),
                           type_convert<dstDataType>{}(beta),
                           static_cast<dstDataType* const __restrict__>(p_dst_global),
                           static_cast<dstDataType* const __restrict__>(ws_buf1_global),
                           static_cast<int* const __restrict__>(ws_buf2_global),
                           static_cast<int* const __restrict__>(indices_global));
		 break; 
            case ReductionMethod_t::MultiBlock:
                 this->Run_MultiBlock<true, true, src2dDescType, dst1dDescType>(src2dDesc, dst1dDesc,
                           type_convert<srcDataType>{}(alpha),
                           static_cast<const srcDataType* const __restrict__>(p_src_global),
                           type_convert<dstDataType>{}(beta),
                           static_cast<dstDataType* const __restrict__>(p_dst_global),
                           static_cast<dstDataType* const __restrict__>(ws_buf1_global),
                           static_cast<int* const __restrict__>(ws_buf2_global),
                           static_cast<int* const __restrict__>(indices_global));
		 break; 
	}; 
    };

    template <typename src2dDescType, typename dst1dDescType>
    __device__ void Run_2(const src2dDescType &src2dDesc, const dst1dDescType &dst1dDesc, 
		                 float alpha,
                                 const void* const __restrict__ p_src_global,
                                 float beta,
                                 void* const __restrict__ p_dst_global,
                                 void* const __restrict__ ws_buf1_global,
                                 long ws_buf2_bytes_offset,
                                 void* const __restrict__ indices_global) const
    {
        (void)p_src_global; // unused

        void* const ws_buf2_global = ws_buf2_bytes_offset > 0 ? static_cast<void*>(static_cast<char*>(ws_buf1_global) + ws_buf2_bytes_offset) : nullptr;

        if ( hipBlockIdx_x == 0 && hipThreadIdx_x == 0 ) 
	     printf("Call Run_2, reduceImpl=%d\n", this->reduceImpl); 

        switch (this->reduceImpl) {
            case ReductionMethod_t::DirectThreadWise:
                 this->Run_DirectThreadWise<false, true, src2dDescType, dst1dDescType>(src2dDesc, dst1dDesc,
                           type_convert<srcDataType>{}(alpha),
                           static_cast<const srcDataType* const __restrict__>(ws_buf1_global),
                           type_convert<dstDataType>{}(beta),
                           static_cast<dstDataType* const __restrict__>(p_dst_global),
                           static_cast<dstDataType* const __restrict__>(nullptr),
                           static_cast<int* const __restrict__>(ws_buf2_global),
                           static_cast<int* const __restrict__>(indices_global));
		 break; 
	    case ReductionMethod_t::DirectWarpWise:
                 this->Run_DirectWarpWise<false, true, src2dDescType, dst1dDescType>(src2dDesc, dst1dDesc,
                           type_convert<srcDataType>{}(alpha),
                           static_cast<const srcDataType* const __restrict__>(ws_buf1_global),
                           type_convert<dstDataType>{}(beta),
                           static_cast<dstDataType* const __restrict__>(p_dst_global),
                           static_cast<dstDataType* const __restrict__>(nullptr),
                           static_cast<int* const __restrict__>(ws_buf2_global),
                           static_cast<int* const __restrict__>(indices_global));
		 break; 
	    case ReductionMethod_t::BlockWise:
                 this->Run_BlockWise<false, true, src2dDescType, dst1dDescType>(src2dDesc, dst1dDesc,
                           type_convert<srcDataType>{}(alpha),
                           static_cast<const srcDataType* const __restrict__>(ws_buf1_global),
                           type_convert<dstDataType>{}(beta),
                           static_cast<dstDataType* const __restrict__>(p_dst_global),
                           static_cast<dstDataType* const __restrict__>(nullptr),
                           static_cast<int* const __restrict__>(ws_buf2_global),
                           static_cast<int* const __restrict__>(indices_global));
		 break; 
	    default: 
		 break; 
        }; 

    };

    ReductionMethod_t reduceImpl; 
    int origReduceLen; 
    int BlkGroupSize; 
};

} // namespace ck
#endif


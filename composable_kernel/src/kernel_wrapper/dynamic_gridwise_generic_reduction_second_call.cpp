#define  HIP_EANBLE_PRINTF

#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"
#include "dynamic_gridwise_generic_reduction_method_chooser.hpp"
#include "dynamic_gridwise_generic_reduction_wrapper_common.hpp"

using namespace ck;

using srcDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_DST_DATATYPE)>::type;
using compType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t BlockSize = CK_PARAM_BLOCKSIZE; // tunable

constexpr index_t srcDims = CK_PARAM_IN_DIMS; 
constexpr index_t dstDims = CK_PARAM_OUT_DIMS; 

using toReduceDims  = Sequence<CK_PARAM_TOREDUCE_DIMS>;
using invariantDims = Sequence<CK_PARAM_INVARIANT_DIMS>;  // this could be empty

constexpr ReduceTensorOp_t op          = get_reduce_op<CK_PARAM_REDUCE_OP>::op;
constexpr NanPropagation_t nanPropaOpt = CK_PARAM_NAN_PROPAGATE == 0
                                             ? NanPropagation_t::NOT_PROPAGATE_NAN
                                             : NanPropagation_t::PROPAGATE_NAN;
constexpr ReduceTensorIndices_t reduceIndicesOpt = CK_PARAM_REDUCE_INDICES == 0
                                                       ? ReduceTensorIndices_t::NO_INDICES
                                                       : ReduceTensorIndices_t::FLATTENED_INDICES;

constexpr index_t GredThreadBufferLength       = CK_PARAM_THREAD_BUFFER_LENGTH;        // tunable
constexpr index_t GredAccessesPerThreadInBlock = CK_PARAM_ACCESSES_PER_THREAD_INBLOCK; // tunable
constexpr index_t GredAccessesPerThreadInWarp  = CK_PARAM_ACCESSES_PER_THREAD_INWARP;  // tunable

////////////////////////////////////////////////////////////////////////////////////////
using specDims = typename sequence_merge<invariantDims, toReduceDims>::type;

static_assert(is_valid_sequence_map<specDims>::value && specDims::Size() == srcDims, "Wrong invariant and/or toReduce dimensions!");

// The number of invariant dimensions can be zero if all dimension are to be reduced
static_assert(invariantDims::Size() > 0 || dstDims == 1, "If all source dimensions are reduced, the dest should have only one dimension !!");

constexpr bool reduceAllDims = (invariantDims::Size() == 0) ? true : false; 

extern "C" __global__ void gridwise_generic_reduce_2_prepare(int reduceImpl2, int GridSize, int BlkGroupSize, 
	                                                     const int * __restrict__ srcLengths, const int *srcStrides, const int *dstLengths, const int *dstStrides, 
		                                             void *p_src2dDesc, void *p_dst1dDesc, bool *p_src_use_padding, bool *p_dst_use_padding)
{
      const auto tupleDstLengths = make_tuple_from_array(dstLengths, Number<dstDims>{});
      const auto tupleDstStrides = make_tuple_from_array(dstStrides, Number<dstDims>{}); 

      const auto dstDesc = make_dynamic_naive_tensor_descriptor_v2(tupleDstLengths, tupleDstStrides);

      const auto one_dim_dstDesc = transform_dynamic_tensor_descriptor(
                                                             dstDesc,
                                                             make_tuple(make_merge_transform(tupleDstLengths)),
                                                             make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                             make_tuple(Sequence<0>{}));

      const index_t invariantLen = one_dim_dstDesc.GetLength(Number<0>{}); 
      const index_t toReduceLen  = BlkGroupSize;

      const auto workspace_2d_desc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(invariantLen, toReduceLen));	

      gridwise_generic_reduce_pad_and_store(static_cast<ReductionMethod_t>(reduceImpl2), GridSize, 0, workspace_2d_desc, one_dim_dstDesc, p_src2dDesc, p_dst1dDesc, p_src_use_padding, p_dst_use_padding); 
};

extern "C" __global__ void gridwise_generic_reduce_2(int reduceImpl2, int origReduceLen, const void CONSTANT *p_src2dDesc, const void CONSTANT *p_dst1dDesc, 
		                                     const bool *p_src_use_padding, const bool *p_dst_use_padding,
		                                     float alpha,
                                                     const void* __restrict__ p_src_global,
                                                     float beta,
                                                     void* __restrict__ p_dst_global,
                                                     void* __restrict__ ws_buf1_global,
                                                     size_t ws_buf2_bytes_offset,
                                                     void* __restrict__ indices_global)
{
    constexpr auto ref_tupleDstLengths = make_tuple_from_seq(typename uniform_sequence_gen<dstDims, 8>::type{}); 
    constexpr auto ref_dstDesc = make_dynamic_naive_tensor_descriptor_v2(ref_tupleDstLengths, ref_tupleDstLengths); 

    constexpr auto ref_dst1dDesc = transform_dynamic_tensor_descriptor(
                                                     ref_dstDesc,
                                                     make_tuple(make_merge_transform(ref_tupleDstLengths)),
                                                     make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                     make_tuple(Sequence<0>{}));

    constexpr index_t ref_invariantLen = ref_dst1dDesc.GetLength(Number<0>{});
    constexpr index_t ref_toReduceLen  = 8;

    constexpr auto ref_src2dDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(ref_invariantLen, ref_toReduceLen));

    using refType_src2dDesc = decltype( ref_src2dDesc ); 
    using refType_dst1dDesc = decltype( ref_dst1dDesc ); 
    
    // used by the DirectThreadWise and DirectWarpWise method
    using refType_src2dDesc_padded_12 = decltype( transform_dynamic_tensor_descriptor(
                                                                    ref_src2dDesc,
                                                                    make_tuple(make_pad_transform(ref_invariantLen, 0, 2), make_pad_transform(ref_toReduceLen, 0, 2)),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{})) );
    
    // used by the BlockWise and MultiBlock method
    using refType_src2dDesc_padded_34 = decltype( transform_dynamic_tensor_descriptor(
                                                                    ref_src2dDesc,
                                                                    make_tuple(make_pass_through_transform(ref_invariantLen), make_pad_transform(ref_toReduceLen, 0, 2)),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{})) );

    using refType_dst1dDesc_padded = decltype( transform_dynamic_tensor_descriptor(
                                                                 ref_dst1dDesc,
                                                                 make_tuple(make_pad_transform(ref_invariantLen, 0, 2)),
                                                                 make_tuple(Sequence<0>{}),
                                                                 make_tuple(Sequence<0>{})) );
    const bool src_use_padding = *p_src_use_padding; 
    const bool dst_use_padding = *p_dst_use_padding; 

    const auto gridwise_2d_reduce = Gridwise2dReduction<BlockSize,
                                                     srcDataType,
                                                     dstDataType,
                                                     compType,
                                                     static_cast<index_t>(op),
                                                     static_cast<index_t>(nanPropaOpt),
                                                     static_cast<index_t>(reduceIndicesOpt),
                                                     GredThreadBufferLength,
                                                     GredAccessesPerThreadInBlock,
                                                     GredAccessesPerThreadInWarp>(reduceImpl2, origReduceLen, 0);

    if ( static_cast<ReductionMethod_t>(reduceImpl2) == ReductionMethod_t::DirectThreadWise || static_cast<ReductionMethod_t>(reduceImpl2) == ReductionMethod_t::DirectWarpWise) {
         if ( src_use_padding && dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_12 *>((const void *)p_src2dDesc); 
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc); 

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
         }
         else if ( src_use_padding && !dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_12 *>((const void *)p_src2dDesc); 
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc *>((const void *)p_dst1dDesc); 

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
	 }
	 else if ( !src_use_padding && dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc *>((const void *)p_src2dDesc); 
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc); 

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
	 }
	 else if ( !src_use_padding && !dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc *>((const void *)p_src2dDesc); 
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc *>((const void *)p_dst1dDesc); 

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
	 }; 
    } 
    else if ( static_cast<ReductionMethod_t>(reduceImpl2) == ReductionMethod_t::BlockWise ) { 
         if ( src_use_padding && dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_34 *>((const void *)p_src2dDesc);
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
         }
         else if ( src_use_padding && !dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_34 *>((const void *)p_src2dDesc);
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc *>((const void *)p_dst1dDesc);

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
         }
         else if ( !src_use_padding && dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc *>((const void *)p_src2dDesc); 
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
         }
         else if ( !src_use_padding && !dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc *>((const void *)p_src2dDesc);
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc *>((const void *)p_dst1dDesc);

              gridwise_2d_reduce.Run(Number<1>{}, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
         };
    };  
};


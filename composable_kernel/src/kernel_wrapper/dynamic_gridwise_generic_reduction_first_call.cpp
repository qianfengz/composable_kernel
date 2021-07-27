#define  HIP_EANBLE_PRINTF

#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"
#include "dynamic_gridwise_generic_reduction.hpp"
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

extern "C" __global__ void gridwise_generic_reduce_1_prepare(int reduceImpl, int GridSize, int BlkGroupSize,  
		                                             const index_t * __restrict__ srcLengths, const index_t *srcStrides, const index_t *dstLengths, const index_t *dstStrides, 
		                                             void *p_src2dDesc, void *p_dst1dDesc, bool *p_src_use_padding, bool *p_dst_use_padding)
{
     const auto tupleSrcLengths = make_tuple_from_array(srcLengths, Number<srcDims>{});
     const auto tupleSrcStrides = make_tuple_from_array(srcStrides, Number<srcDims>{});
     const auto tupleDstLengths = make_tuple_from_array(dstLengths, Number<dstDims>{});
     const auto tupleDstStrides = make_tuple_from_array(dstStrides, Number<dstDims>{});

     const auto srcDesc = make_dynamic_naive_tensor_descriptor_v2(tupleSrcLengths, tupleSrcStrides);
     const auto dstDesc = make_dynamic_naive_tensor_descriptor_v2(tupleDstLengths, tupleDstStrides);

#ifndef CK_REDUCE_ALL_DIMS     
           // for re-ordering the tensor dimensions
           using lowDimSeq  = typename sequence_merge<invariantDims, toReduceDims>::type;
           using highDimSeq = typename arithmetic_sequence_gen<0, srcDims, 1>::type;

           const auto toReduceDimLengths  = make_tuple_from_array_and_index_seq(srcLengths, toReduceDims{});
           const auto invariantDimLengths = make_tuple_from_array_and_index_seq(srcLengths, invariantDims{});

           // construct the reordered tensor descriptor according to the srcMode and dstMode mapping
           const auto reordered_srcDesc = transform_dynamic_tensor_descriptor(
                                                             srcDesc,
                                                             make_passthrough_tuple_from_array_and_index_seq(srcLengths, lowDimSeq{}),
                                                             make_dimensions_tuple(lowDimSeq{}),
                                                             make_dimensions_tuple(highDimSeq{}));

           const auto two_dim_srcDesc = transform_dynamic_tensor_descriptor(
                                                           reordered_srcDesc,
                                                           make_tuple(make_merge_transform(invariantDimLengths), make_merge_transform(toReduceDimLengths)),
                                                           make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{},
                                                                      typename arithmetic_sequence_gen<dstDims, srcDims, 1>::type{}),
                                                           make_tuple(Sequence<0>{}, Sequence<1>{}));

           const auto one_dim_dstDesc = transform_dynamic_tensor_descriptor(
                                                           dstDesc,
                                                           make_tuple(make_merge_transform(tupleDstLengths)),
                                                           make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                           make_tuple(Sequence<0>{}));

           gridwise_generic_reduce_pad_and_store(static_cast<ReductionMethod_t>(reduceImpl), GridSize, BlkGroupSize, two_dim_srcDesc, one_dim_dstDesc, p_src2dDesc, p_dst1dDesc, p_src_use_padding, p_dst_use_padding);
#else
           const auto one_dim_srcDesc = transform_dynamic_tensor_descriptor(
                                                           srcDesc,
                                                           make_tuple(make_merge_transform(tupleSrcLengths)),
                                                           make_tuple(typename arithmetic_sequence_gen<0, srcDims, 1>::type{}),
                                                           make_tuple(Sequence<0>{}));

           const auto two_dim_srcDesc = transform_dynamic_tensor_descriptor(
                                                           one_dim_srcDesc,
                                                           make_tuple(make_unmerge_transform(make_tuple(1, one_dim_srcDesc.GetLength(Number<0>{})))),
                                                           make_tuple(Sequence<0>{}),
                                                           make_tuple(Sequence<0, 1>{}));

           const auto one_dim_dstDesc = transform_dynamic_tensor_descriptor(
                                                           dstDesc,
                                                           make_tuple(make_merge_transform(tupleDstLengths)),
                                                           make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                           make_tuple(Sequence<0>{}));

           gridwise_generic_reduce_pad_and_store(static_cast<ReductionMethod_t>(reduceImpl), GridSize, BlkGroupSize, two_dim_srcDesc, one_dim_dstDesc, p_src2dDesc, p_dst1dDesc, p_src_use_padding, p_dst_use_padding);
#endif

}; 

extern "C" __global__ void gridwise_generic_reduce_1(int reduceImpl, int origReduceLen, int BlkGroupSize, const void __CONSTANT__ *p_src2dDesc, const void __CONSTANT__ *p_dst1dDesc,
	                                             const bool *p_src_use_padding, const bool *p_dst_use_padding,
		                                     float alpha,
                                                     const void* __restrict__ p_src_global,
                                                     float beta,
                                                     void* __restrict__ p_dst_global,
                                                     void* __restrict__ ws_buf1_global,
                                                     size_t ws_buf2_bytes_offset,
                                                     void* __restrict__ indices_global)
{
      using refType_src2dDesc = typename get_ref_desc_types<reduceAllDims, srcDims, dstDims, invariantDims, toReduceDims>::refType_src2dDesc; 
      using refType_dst1dDesc = typename get_ref_desc_types<reduceAllDims, srcDims, dstDims, invariantDims, toReduceDims>::refType_dst1dDesc; 
      using refType_src2dDesc_padded_12 = typename get_ref_desc_types<reduceAllDims, srcDims, dstDims, invariantDims, toReduceDims>::refType_src2dDesc_padded_12; 
      using refType_src2dDesc_padded_34 = typename get_ref_desc_types<reduceAllDims, srcDims, dstDims, invariantDims, toReduceDims>::refType_src2dDesc_padded_34; 
      using refType_dst1dDesc_padded = typename get_ref_desc_types<reduceAllDims, srcDims, dstDims, invariantDims, toReduceDims>::refType_dst1dDesc_padded; 

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
                                                       GredAccessesPerThreadInWarp>(reduceImpl, origReduceLen, BlkGroupSize);

      if ( static_cast<ReductionMethod_t>(reduceImpl) == ReductionMethod_t::DirectThreadWise || static_cast<ReductionMethod_t>(reduceImpl) == ReductionMethod_t::DirectWarpWise) {
           if ( src_use_padding && dst_use_padding ) {
                 const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_12 *>((const void *)p_src2dDesc);
                 const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

                 gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc,
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

                     gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc, 
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

                     gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc, 
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

                     gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc, 
                                         alpha,
                                         const_cast<const void* const __restrict__>(p_src_global),
                                         beta,
                                         const_cast<void* const __restrict__>(p_dst_global),
                                         const_cast<void* const __restrict__>(ws_buf1_global),
                                         ws_buf2_bytes_offset,
                                         const_cast<void* const __restrict__>(indices_global));
           };
      }
      else if ( static_cast<ReductionMethod_t>(reduceImpl) == ReductionMethod_t::BlockWise || static_cast<ReductionMethod_t>(reduceImpl) == ReductionMethod_t::MultiBlock ) {
                if ( src_use_padding && dst_use_padding ) {
                     const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_34 *>((const void *)p_src2dDesc);
                     const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

                     gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc,  
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

                          gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc, 
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

                          gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc, 
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

                          gridwise_2d_reduce.Run(Number<0>{}, src2dDesc, dst1dDesc,  
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


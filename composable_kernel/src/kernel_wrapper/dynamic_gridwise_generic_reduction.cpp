#include "config.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "tensor_descriptor_helper.hpp"
#include "reduction_common.hpp"
#include "dynamic_gridwise_generic_reduction.hpp"

using namespace ck;

template <char tid>
struct get_type_from_type_id
{
    using type = float;
};

template <>
struct get_type_from_type_id<'H'>
{
    using type = half_t;
};

template <>
struct get_type_from_type_id<'F'>
{
    using type = float;
};

template <>
struct get_type_from_type_id<'D'>
{
    using type = double;
};

template <index_t persistentID>
struct get_reduce_op // any other ID
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<656868> // 'A' * 10000 + 'D' * 100 + 'D'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::ADD;
};

template <>
struct get_reduce_op<778576> // 'M' * 10000 + 'U' * 100 + 'L'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MUL;
};

template <>
struct get_reduce_op<777378> // 'M' * 10000 + 'I' * 100 + 'N'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MIN;
};

template <>
struct get_reduce_op<776588> // 'M' * 10000 + 'A' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::MAX;
};

template <>
struct get_reduce_op<657788> // 'A' * 10000 + 'M' * 100 + 'X'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AMAX;
};

template <>
struct get_reduce_op<658671> // 'A' * 10000 + 'V' * 100 + 'G'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::AVG;
};

template <>
struct get_reduce_op<788201> // 'N' * 10000 + 'R' * 100 + '1'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM1;
};

template <>
struct get_reduce_op<788202> // 'N' * 10000 + 'R' * 100 + '2'
{
    static constexpr ReduceTensorOp_t op = ReduceTensorOp_t::NORM2;
};

using srcDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_SRC_DATATYPE)>::type;
using dstDataType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_DST_DATATYPE)>::type;
using compType = typename get_type_from_type_id<static_cast<char>(CK_PARAM_REDUCE_COMPTYPE)>::type;

constexpr index_t blockSize = CK_PARAM_BLOCKSIZE; // tunable

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


template <index_t... Ns>
static auto make_tuple_from_array_and_index_seq(const size_t *lengths, Sequence<Ns...>)
{
    return make_tuple(lengths[Ns]...);
};

template <index_t arraySize>
static auto make_tuple_from_array(const size_t *lengths, Number<arraySize>)
{
   static_assert(arraySize >=1 && arraySize <= 6, "The tensor should have 1 to 6 dimensions"); 

   constexpr auto index_seq = typename arithmetic_sequence_gen<0, arraySize, 1>::type{}; 

   return make_tuple_from_array_and_index_seq(lengths, index_seq); 
}; 

template <index_t... Ns>
static auto make_passthrough_tuple_from_array_and_index_seq(const size_t *lengths, Sequence<Ns...>)
{
    return make_tuple(make_passthrough_transform(lengths[Ns])...);
};

template <index_t... Ids>
static constexpr auto make_tuple_from_seq(Sequence<Ids...>)
{
    return make_tuple(Ids...);
};

template <index_t... Is>
static constexpr auto make_passthrough_tuple_from_seq(Sequence<Is...>)
{
    return make_tuple(make_passthrough_transform(Is)...);
};

template <typename src2dDescType, typename dst1dDescType>
static inline void gridwise_generic_reduce_pad_and_store(ReductionMethod_t reduceImpl, size_t GridSize, int BlkGroupSize, src2dDescType &src2dDesc, dst1dDescType &dst1dDesc,
	                                                 void *p_src2dDesc, void *p_dst1dDesc, bool *p_src_use_padding, bool *p_dst_use_padding)
{
     const auto invariantLen = src2dDesc.GetLength(Number<0>{}); 
     const auto toReduceLen = src2dDesc.GeLength(Number<1>{}); 

     switch (reduceImpl) {
         case ReductionMethod_t::DirectThreadWise:
              {	
                  constexpr auto copySliceLen = GredThreadBufferLength;
                  const bool src_need_padding = (invariantLen < GridSize * BlockSize || toReduceLen % copySliceLen > 0) ? true : false;

                  if ( src_need_padding ) {
                       const auto srcPad1 = GridSize * BlockSize - invariantLen;
                       const auto srcPad2 = ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;
                       const auto src2dDesc_2 = transform_dynamic_tensor_descriptor(
                                                                          src2dDesc,
                                                                          make_tuple(make_pad_transform(invariantLen, 0, srcPad1), make_pad_transform(toReduceLen, 0, srcPad2)),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}));
                       *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2; 
                       *p_src_use_padding = true; 
                  }
		  else {
                       *static_cast<decltype(src2dDesc)*>(p_src2dDesc) = src2dDesc; 
		       *p_src_use_padding = false;
		  }; 

                  const auto dst_need_padding = (invariantLen < GridSize * BlockSize) ? true : false;

                  if ( dst_need_padding ) {
                       const auto dstPad = GridSize * BlockSize - invariantLen;
                       const auto dst1dDesc_2 = transform_dynamic_tensor_descriptor(
                                                                          dst1dDesc,
                                                                          make_tuple(make_pad_transform(invariantLen, 0, dstPad)),
                                                                          make_tuple(Sequence<0>{}),
                                                                          make_tuple(Sequence<0>{}));
		       *static_cast<decltype(dst1dDesc_2)*>(p_dst1dDesc) = dst1dDesc_2; 
		       *p_dst_use_padding = true; 
                  }
		  else {
		       *static_cast<decltype(dst1dDec)*>(p_dst1dDesc) = dst1dDesc; 
		       *p_dst_use_padding = false; 
		  }; 

	      }; 	 
	      break; 
	 case ReductionMethod_t::DirectWarpWise:
              {
                  constexpr auto copySliceLen = warpSize * GredAccessesPerThreadInWarp;
                  const bool src_need_padding = (invariantLen < GridSize * BlockSize / warpSize || toReduceLen % copySliceLen > 0) ? true : false;

                  if ( src_need_padding ) {
                       const auto srcPad1 = GridSize * BlockSize / warpSize - invariantLen;
                       const auto srcPad2 = ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

                       const auto src2dDesc_2 = transform_dynamic_tensor_descriptor(
                                                                          src2dDesc,
                                                                          make_tuple(make_pad_transform(invariantLen, 0, srcPad1), make_pad_transform(toReduceLen, 0, srcPad2)),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}));
                       *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2; 
                       *p_src_use_padding = true; 
                       
                  }
		  else {
                       *static_cast<decltype(src2dDesc)*>(p_src2dDesc) = src2dDesc; 
		       *p_src_use_padding = false;
		  }; 

                  const auto dst_need_padding = (invariantLen < GridSize * BlockSize / warpSize) ? true : false;

                  if ( dst_need_padding ) {
                       const auto dstPad = GridSize * BlockSize / warpSize - invariantLen;
                       const auto dst1dDesc_2 = transform_dynamic_tensor_descriptor(
                                                                          dst1dDesc,
                                                                          make_tuple(make_pad_transform(invariantLen, 0, dstPad)),
                                                                          make_tuple(Sequence<0>{}),
                                                                          make_tuple(Sequence<0>{}));
		       *static_cast<decltype(dst1dDesc_2)*>(p_dst1dDesc) = dst1dDesc_2; 
		       *p_dst_use_padding = true; 
		  }
		  else {
		       *static_cast<decltype(dst1dDec)*>(p_dst1dDesc) = dst1dDesc; 
		       *p_dst_use_padding = false; 
		  }; 
	      };
	      break; 
	 case ReductionMethod_t::BlockWise:
              {
                  constexpr auto copySliceLen = BlockSize * GredAccessesPerThreadInBlock;
                  const bool src_need_padding = (toReduceLen % copySliceLen > 0) ? true : false;

                  if ( src_need_padding ) {
                       const auto srcPad = ((toReduceLen + copySliceLen - 1) / copySliceLen) * copySliceLen - toReduceLen;

                       const auto src2dDesc_2 = transform_dynamic_tensor_descriptor(
                                                                          src2dDesc,
                                                                          make_tuple(make_passthrough_transform(invariantLen), make_pad_transform(toReduceLen>, 0, srcPad)),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}));
                       *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2; 
                       *p_src_use_padding = true; 
                  }
		  else {
                       *static_cast<decltype(src2dDesc)*>(p_src2dDesc) = src2dDesc; 
		       *p_src_use_padding = false;
		  }; 

		  *static_cast<decltype(dst1dDec)*>(p_dst1dDesc) = dst1dDesc; 
		  *p_dst_use_padding = false; 
	      }; 
	      break; 
	 case ReductionMethod_t::MultiBlock:
              {
                  const auto copySliceLen = BlockSize * GredAccessesPerThreadInBlock;
                  const index_t reduceSizePerBlock = (((toReduceLen + BlkGroupSize - 1) / BlkGroupSize + copySliceLen - 1) / copySliceLen) * copySliceLen;
                  const bool src_need_padding = (toReduceLen < reduceSizePerBlock * BlkGroupSize) ? true : false;

                  if ( src_need_padding ) {
                       const auto srcPad = reduceSizePerBlock * BlkGroupSize - toReduceLen;

                       const auto src2dDesc_2 = transform_dynamic_tensor_descriptor(
                                                                          src2dDesc,
                                                                          make_tuple(make_passthrough_transform(invariantLen), make_pad_transform(toReduceLen, 0, srcPad)),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                          make_tuple(Sequence<0>{}, Sequence<1>{}));	      
                       *static_cast<decltype(src2dDesc_2)*>(p_src2dDesc) = src2dDesc_2; 
                       *p_src_use_padding = true; 
                  }
		  else {
                       *static_cast<decltype(src2dDesc)*>(p_src2dDesc) = src2dDesc; 
		       *p_src_use_padding = false;
		  }; 

		  *static_cast<decltype(dst1dDec)*>(p_dst1dDesc) = dst1dDesc; 
		  *p_dst_use_padding = false; 
              };	      
	      break;
     }; 
}; 

extern "C" __global__ void gridwise_generic_reduce_1_prepare(int reduceImpl, size_t GridSize, 
		                                             const size_t *srcLengths, const size_t *srcStrides, const size_t *dstLengths, const size_t *dstStrides, 
		                                             void *p_src2dDesc, void *p_dst1dDesc, bool *p_src_use_padding, bool *p_dst_use_padding)
{
     const auto tupleSrcLengths = make_tuple_from_array(srcLengths, Number<srcDims>{});
     const auto tupleSrcStrides = make_tuple_from_array(srcStrides, Number<srcDims>{});
     const auto tupleDstLengths = make_tuple_from_array(dstLengths, Number<dstDims>{});
     const auto tupleDstStrides = make_tuple_from_array(dstStrides, Number<dstDims>{});

     const srcDesc = make_dynamic_naive_tensor_descriptor_v2(tupleSrcLengths, tupleSrcStrides);
     const dstDesc = make_dynamic_naive_tensor_descriptor_v2(tupleDstLengths, tupleDstStrides);

     static_if<!reduceAllDims>{}([&](auto) { // not all dimensions are to be reduced
            toReduceDimLengths  = make_tuple_from_array_and_index_seq(srcLengths, toReduceDims{}));
            invariantDimLengths = make_tuple_from_array_and_index_seq(srcLengths, invariantDims{}))

            // for re-ordering the tensor dimensions
            using lowDimSeq  = typename sequence_merge<invariantDims, toReduceDims>::type;
            using highDimSeq = typename arithmetic_sequence_gen<0, srcDims, 1>::type;

            // construct the reordered tensor descriptor according to the srcMode and dstMode mapping
            const auto reordered_srcDesc = transform_dynamic_tensor_descriptor(
                                                             srcDesc,
                                                             make_passthrough_tuple_from_array_and_index_seq(srcLengths, lowDimSeq{}),
                                                             make_tuple_from_seq(lowDimSeq{}),
                                                             make_tuple_from_seq(highDimSeq{}));
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

            gridwise_generic_reduce_padding_and_store(static_cast<ReductionMethod_t>(reduceImpl), GridSize, two_dim_srcDesc, one_dim_dstDesc, p_src2dDesc, p_dst1dDesc, p_src_use_padding, p_dst_use_padding); 
     }).Else([&](auto) { // All dimensions are to be reduced
            const auto one_dim_srcDesc = transform_dynamic_tensor_descriptor(
                                                           srcDesc,
                                                           make_tuple(make_merge_transform(tupleSrcLengths)),
                                                           make_tuple(typename arithmetic_sequence_gen<0, srcDims, 1>::type{}),
                                                           make_tuple(Sequence<0>{}));

            const auto two_dim_srcDesc = transform_dynamic_tensor_descriptor(
	                                                   one_dim_srcDesc,
                                                           make_tuple(make_unmerge_transform(make_tuple(1, one_dim_srcDesc.GetLengths()[0])),
                                                           make_tuple(Sequence<0>{}),
                                                           make_tuple(Sequence<0, 1>{}));

            const auto one_dim_dstDesc = transform_dynamic_tensor_descriptor(
                                                           dstDesc,
                                                           make_tuple(make_merge_transform(tupleDstLengths)),
                                                           make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                           make_tuple(Sequence<0>{}));

            gridwise_generic_reduce_pad_and_store(static_cast<ReductionMethod_t>(reduceImpl), GridSize, two_dim_srcDesc, one_dim_dstDesc, p_src2dDesc, p_dst1dDesc, p_src_use_padding, p_dst_use_padding); 
        });	    
}; 

extern "C" __global__ void gridwise_generic_reduce_2_prepare(int reduceImpl2, size_t GridSize,
	                                                     const size_t *srcLengths, const size_t *srcStrides, const size_t *dstLengths, const size_t *dstStrides, 
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

      const index_t invariantLen = one_dim_dstDesc.GetLengths()[0]; 
      const index_t toReduceLen  = BlkGroupSize;

      const auto workspace_2d_desc = make_dynamic_native_tensor_descriptor_packed_v2(invariantLen, toReduceLen);	

      gridwise_generic_reduce_pad_and_store(static_cast<ReductionMethod_t>(reduceImpl2), GridSize, workspace_2d_desc, one_dim_dstDesc, p_src2dDesc, p_dst1dDesc, p_src_use_padding, p_dst_use_padding); 
};

template <bool reduceAllDims, index_t srcDims, index_t dstDims, typename invariantDims, typename toReduceDims>
struct get_ref_desc_types; 

template <index_t srcDims, index_t dstDims, typename invariantDims, typename toReduceDims>
struct get_ref_desc_types<true, index_t srcDims, index_t dstDims, typename invariantDims, typename toReduceDims>
{
      static constexpr auto ref_toReduceDimLengths = typename uniform_sequence_gen<toReduceDims::Size(), 8>::type{};
      static constexpr auto ref_invariantDimLengths = typename uniform_sequence_gen<invairantDims::Size(), 8>::type{};

      // for re-ordering the tensor dimensions
      using lowDimSeq  = typename sequence_merge<invariantDims, toReduceDims>::type;
      using highDimSeq = typename arithmetic_sequence_gen<0, srcDims, 1>::type;

      static constexpr auto ref_srcLengths = typename uniform_sequence_gen<srcDims, 8>::type{};
      static constexpr auto ref_dstLengths = typename uniform_sequence_gen<dstDims, 8>::type{};

      static constexpr auto ref_srcDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple_from_seq(ref_srcLengths));
      static constexpr auto ref_dstDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple_from_seq(ref_dstLengths));

      // construct the reordered tensor descriptor according to the srcMode and dstMode mapping
      static constexpr auto ref_reordered_srcDesc = transform_dynamic_tensor_descriptor(
                                                               ref_srcDesc,
                                                               make_passthrough_tuple_from_seq(ref_srcLengths),
                                                               make_tuple_from_seq(lowDimSeq{}),
                                                               make_tuple_from_seq(highDimSeq{}));
      static constexpr auto ref_src2dDesc = transform_dynamic_tensor_descriptor(
                                                       ref_reordered_srcDesc,
                                                       make_tuple(make_merge_transform(make_tuple_from_seq(ref_invariantDimLengths)), make_merge_transform(make_tuple_from_seq(ref_toReduceDimLengths))),
                                                       make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{},
                                                                  typename arithmetic_sequence_gen<dstDims, srcDims, 1>::type{}),
                                                       make_tuple(Sequence<0>{}, Sequence<1>{}));

      static constexpr auto ref_dst1dDesc = transform_dynamic_tensor_descriptor(
                                                       ref_dstDesc,
                                                       make_tuple(make_merge_transform(make_tuple_from_seq(ref_dstLengths))),
                                                       make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                       make_tuple(Sequence<0>{}));

      using refType_src2dDesc = decltype( ref_src2dDesc );
      using refType_dst1dDesc = decltype( ref_dst1dDesc );
}; 

template <index_t srcDims, index_t dstDims, typename invariantDims, typename toReduceDims>
struct get_ref_desc_types<false, index_dstDims, typename invariantDims, typename toReduceDims>
{
      static constexpr auto ref_srcLengths = typename uniform_sequence_gen<srcDims, 8>::type{};
      static constexpr auto ref_dstLengths = typename uniform_sequence_gen<dstDims, 1>::type{};

      static constexpr auto ref_srcDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple_from_seq(ref_srcLengths));
      static constexpr auto ref_dstDesc = make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple_from_seq(ref_dstLengths));

      static constexpr auto ref_one_dim_srcDesc = transform_dynamic_tensor_descriptor(
                                                                    ref_srcDesc,
                                                                    make_tuple(make_merge_transform(make_tuple_from_seq(ref_srcLengths))),
                                                                    make_tuple(typename arithmetic_sequence_gen<0, srcDims, 1>::type{}),
                                                                    make_tuple(Sequence<0>{}));

      static constexpr auto ref_src2dDesc = transform_dynamic_tensor_descriptor(
                                                              ref_one_dim_srcDesc,
                                                              make_tuple(make_unmerge_transform(make_tuple(1, ref_one_dim_srcDesc.GetLengths()[0]))),
                                                              make_tuple(Sequence<0>{}),
                                                              make_tuple(Sequence<0, 1>{}));

      static constexpr auto ref_dst1dDesc = transform_dynamic_tensor_descriptor(
                                                              ref_dstDesc,
                                                              make_tuple(make_merge_transform(make_tuple_from_seq(ref_dstLengths))),
                                                              make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                              make_tuple(Sequence<0>{}));

      using refType_src2dDesc = decltype( ref_src2dDesc );
      using refType_dst1dDesc = decltype( ref_dst1dDesc );
};

extern "C" __global__ void gridwise_generic_reduce_1(int reduceImpl, int origReduceLen, int BlkGroupSize, const void __CONSTANT__ *p_src2dDesc, const void __CONSTANT__ *p_dst1dDesc,
	                                             const bool *p_src_use_padding, const bool *p_dst_use_padding,
		                                     float alpha,
                                                     const void* p_src_global,
                                                     float beta,
                                                     void* p_dst_global,
                                                     void* ws_buf1_global,
                                                     long ws_buf2_bytes_offset,
                                                     void* indices_global)
{
      using refType_src2dDesc = typename get_ref_desc_types<reduceAllDims, srcDims, dstDims, invariantDims, toReduceDims>::refType_src2dDesc; 
      using refType_dst1dDesc = typename get_ref_desc_types<reduceAllDims, srcDims, dstDims, invariantDims, toReduceDims>::refType_dst1dDesc; 

      constexpr auto ref_invariantLen = refType_src2dDesc::GetLengths()[0];
      constexpr auto ref_toReduceLen = refType_src2dDesc::GetLengths()[1];

      // used by the Direct_ThreadWise and Direct_WarpWise method
      using refType_src2dDesc_padded_12 = decltype( transform_dynamic_tensor_descriptor(
                                                                      refType_src2dDesc{},
                                                                      make_tuple(make_pad_transform(ref_invariantLen, 0, 2), make_pad_transform(ref_toReduceLen, 0, 2)),
                                                                      make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                      make_tuple(Sequence<0>{}, Sequence<1>{})) );

      // used by the BlockWise and MultiBlock method
      using refType_src2dDesc_padded_34 = decltype( transform_dynamic_tensor_descriptor(
                                                                      refType_src2dDesc{},
                                                                      make_tuple(make_passthrough_transform(ref_invariantLen), make_pad_transform(ref_toReduceLen, 0, 2)),
                                                                      make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                      make_tuple(Sequence<0>{}, Sequence<1>{})) );

      using refType_dst1dDesc_padded = decltype( transform_dynamic_tensor_descriptor(
                                                                   refType_dst1dDesc{},
                                                                   make_tuple(make_pad_transform(ref_invariantLen, 0, 2)),
                                                                   make_tuple(Sequence<0>{}),
                                                                   make_tuple(Sequence<0>{})) );

      const bool src_use_padding = *p_src_use_padding; 
      const bool dst_use_padding = *p_dst_use_padding; 

      const auto gridwise_reduce = Gridwise2dReduction<blockSize,
                                                       srcDataType,
                                                       dstDataType,
                                                       compType,
                                                       static_cast<index_t>(op),
                                                       static_cast<index_t>(nanPropaOpt),
                                                       static_cast<index_t>(reduceIndicesOpt),
                                                       GredThreadBufferLength,
                                                       GredAccessesPerThreadInBlock,
                                                       GredAccessesPerThreadInWarp>(origReduceLen, BlkGroupSize);

      if ( static_cast<ReductionMethod_t>(reduceImpl) == ReductionMethod_t::Direct_ThreadWise || static_cast<ReductionMethod_t>(reduceImpl) == ReductionMethod_t::Direct_WarpWise) {
           if ( src_use_padding && dst_use_padding ) {
                 const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_12 *>((const void *)p_src2dDesc);
                 const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

                 gridwise_reduce.Run(reduceImpl, src2dDesc, dst1dDesc,
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

                     gridwise_reduce.Run(reduceImpl, src2dDesc, dst1dDesc, 
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

                     gridwise_reduce.Run(reduceImpl, src2dDesc, dst1dDesc, 
                                         alpha,
                                         const_cast<const void* const __restrict__>(p_src_global),
                                         beta,
                                         const_cast<void* const __restrict__>(p_dst_global),
                                         const_cast<void* const __restrict__>(ws_buf1_global),
                                         ws_buf2_bytes_offset,
                                         const_cast<void* const __restrict__>(indices_global));
           }
           else if ( !src_use_padding && !dst_use_padding ) {
                     const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded *>((const void *)p_src2dDesc);
                     const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

                     gridwise_reduce.Run(reduceImpl, src2dDesc, dst1dDesc, 
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

                     gridwise_reduce.Run(reduceImpl, src2dDesc, dst1dDesc,  
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

                          gridwise_reduce.Run(reduceImpl, src2dDesc, dst1dDesc, 
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

                          gridwise_reduce.Run(reduceImpl2, src2dDesc, dst1dDesc, 
                                              alpha,
                                              const_cast<const void* const __restrict__>(p_src_global),
                                              beta,
                                              const_cast<void* const __restrict__>(p_dst_global),
                                              const_cast<void* const __restrict__>(ws_buf1_global),
                                              ws_buf2_bytes_offset,
                                              const_cast<void* const __restrict__>(indices_global));
                }
                else if ( !src_use_padding && !dst_use_padding ) {
                          const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded *>((const void *)p_src2dDesc);
                          const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

                          gridwise_reduce.Run(reduceImpl2, src2dDesc, dst1dDesc,  
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

extern "C" __global__ void gridwise_generic_reduce_2(int reduceImpl2, int origReduceLen, const void __CONSTANT__ *p_src2dDesc, const void __CONSTANT__ *p_dst1dDesc, 
		                                     const bool *p_src_use_padding, const bool *p_dst_use_padding,
		                                     float alpha,
                                                     const void* p_src_global,
                                                     float beta,
                                                     void* p_dst_global,
                                                     void* ws_buf1_global,
                                                     long ws_buf2_bytes_offset,
                                                     void* indices_global)
{
    constexpr auto ref_tupleDstLengths = typename uniform_sequence_gen<dstDims, 8>::type{}; 
    constexpr auto ref_dstDesc = make_dynamic_naive_tensor_descriptor_packed_v2(tupleDstLengths); 

    constexpr auto ref_dst1dDesc = transform_dynamic_tensor_descriptor(
                                                     ref_dstDesc,
                                                     make_tuple(make_merge_transform(ref_tupleDstLengths)),
                                                     make_tuple(typename arithmetic_sequence_gen<0, dstDims, 1>::type{}),
                                                     make_tuple(Sequence<0>{}));

    constexpr index_t ref_invariantLen = ref_dst1dDesc.GetLengths()[0];
    constexpr index_t ref_toReduceLen  = 8;

    constexpr auto ref_src2dDesc = make_dynamic_native_tensor_descriptor_packed_v2(ref_invariantLen, ref_toReduceLen);

    using refType_src2dDesc = decltype( ref_src2dDesc ); 
    using refType_dst1dDesc = decltype( ref_dst1dDesc ); 
    
    // used by the Direct_ThreadWise and Direct_WarpWise method
    using refType_src2dDesc_padded_12 = decltype( transform_dynamic_tensor_descriptor(
                                                                    ref_src2dDesc,
                                                                    make_tuple(make_pad_transform(ref_invariantLen, 0, 2), make_pad_transform(ref_toReduceLen, 0, 2)),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{})) );
    
    // used by the BlockWise and MultiBlock method
    using refType_src2dDesc_padded_34 = decltype( transform_dynamic_tensor_descriptor(
                                                                    ref_src2dDesc,
                                                                    make_tuple(make_passthrough_transform(ref_invariantLen), make_pad_transform(ref_toReduceLength, 0, 2)),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                                                    make_tuple(Sequence<0>{}, Sequence<1>{})) );

    using refType_dst1dDesc_padded = decltype( transform_dynamic_tensor_descriptor(
                                                                 ref_dst1dDesc,
                                                                 make_tuple(make_pad_transform(ref_invariantLen, 0, 2)),
                                                                 make_tuple(Sequence<0>{}),
                                                                 make_tuple(Sequence<0>{})) );
    const bool src_use_padding = *p_src_use_padding; 
    const bool dst_use_padding = *p_dst_use_padding; 

    const auto gridwise_reduce = Gridwise2dReduction<blockSize,
                                                     srcDataType,
                                                     dstDataType,
                                                     compType,
                                                     static_cast<index_t>(op),
                                                     static_cast<index_t>(nanPropaOpt),
                                                     static_cast<index_t>(reduceIndicesOpt),
                                                     GredThreadBufferLength,
                                                     GredAccessesPerThreadInBlock,
                                                     GredAccessesPerThreadInWarp>(origReduceLen, BlkGroupSize);

    if ( static_cast<ReductionMethod_t>(reduceImpl2) == ReductionMethod_t::Direct_ThreadWise || static_cast<ReductionMethod_t>(reduceImpl2) == ReductionMethod_t::Direct_WarpWise) {
         if ( src_use_padding && dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded_12 *>((const void *)p_src2dDesc); 
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc); 

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,  
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

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,  
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

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
	 }
	 else if ( !src_use_padding && !dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded *>((const void *)p_src2dDesc); 
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc); 

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,
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

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,  
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

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,  
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

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,  
                                    alpha,
                                    const_cast<const void* const __restrict__>(p_src_global),
                                    beta,
                                    const_cast<void* const __restrict__>(p_dst_global),
                                    const_cast<void* const __restrict__>(ws_buf1_global),
                                    ws_buf2_bytes_offset,
                                    const_cast<void* const __restrict__>(indices_global));
         }
         else if ( !src_use_padding && !dst_use_padding ) {
              const auto src2dDesc = *reinterpret_cast<const refType_src2dDesc_padded *>((const void *)p_src2dDesc);
              const auto dst1dDesc = *reinterpret_cast<const refType_dst1dDesc_padded *>((const void *)p_dst1dDesc);

              gridwise_reduce.Run_2(reduceImpl2, src2dDesc, dst1dDesc,  
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


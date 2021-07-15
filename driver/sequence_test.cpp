#include "config.hpp"
#include "sequence.hpp"
#include "tuple.hpp"
#include "dynamic_multi_index_transform_helper.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

#include <iostream>

using namespace ck;

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

template <index_t... Ids>
static auto make_passthrough_tuple_from_array_and_index_seq(const size_t *lengths, Sequence<Ids...>)
{
    return make_tuple(make_pass_through_transform(lengths[Ids])...);
};

template <index_t... Ns>
static constexpr auto make_tuple_from_seq(Sequence<Ns...>)
{
    return make_tuple(Ns...);
};

template <index_t... Ns>
static constexpr auto make_dimensions_tuple(Sequence<Ns...>)
{
    return make_tuple(Sequence<Ns>{}...);
};

template <index_t... Ns>
static constexpr auto make_passthrough_tuple_from_seq(Sequence<Ns...>)
{
    return make_tuple(make_pass_through_transform(Ns)...);
};

int main()
{
      constexpr int srcDims = 2; 
      constexpr int dstDims = 1; 
      using invariantDims = Sequence<1>; 
      using toReduceDims = Sequence<0>;

      static constexpr auto ref_toReduceDimLengths = Sequence<64>{};
      static constexpr auto ref_invariantDimLengths = Sequence<30>{};

      // for re-ordering the tensor dimensions
      using lowDimSeq  = typename sequence_merge<invariantDims, toReduceDims>::type;
      using highDimSeq = typename arithmetic_sequence_gen<0, srcDims, 1>::type;

      static constexpr auto ref_srcLengths = Sequence<64, 30>{};  
      static constexpr auto ref_dstLengths = Sequence<30>{};

      static constexpr auto ref_srcStrides = Sequence<30, 1>{}; 
      static constexpr auto ref_dstStrides = Sequence<1>{}; 

      static constexpr auto tuple_ref_srcLengths = make_tuple_from_seq(ref_srcLengths);
      static constexpr auto tuple_ref_dstLengths = make_tuple_from_seq(ref_dstLengths);
      static constexpr auto tuple_ref_srcStrides = make_tuple_from_seq(ref_srcStrides); 
      static constexpr auto tuple_ref_dstStrides = make_tuple_from_seq(ref_dstStrides); 

      std::cout << "tuple_ref_srcLengths: " << tuple_ref_srcLengths[Number<0>{}] << "," << tuple_ref_srcLengths[Number<1>{}] << std::endl; 

      static constexpr auto ref_srcDesc = make_dynamic_naive_tensor_descriptor_v2(tuple_ref_srcLengths, tuple_ref_srcStrides);
      static constexpr auto ref_dstDesc = make_dynamic_naive_tensor_descriptor_v2(tuple_ref_dstLengths, tuple_ref_dstStrides);

      static constexpr auto ref_reordered_srcDesc = transform_dynamic_tensor_descriptor(
                                                               ref_srcDesc,
                                                               make_passthrough_tuple_from_seq(ref_srcLengths),
                                                               make_dimensions_tuple(lowDimSeq{}),
                                                               make_dimensions_tuple(highDimSeq{}));
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

      static constexpr auto ref_invariantLen = ref_src2dDesc.GetLength(Number<0>{});
      static constexpr auto ref_toReduceLen = ref_src2dDesc.GetLength(Number<1>{});

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

      using refType_src2dDesc = decltype( ref_src2dDesc );
      using refType_dst1dDesc = decltype( ref_dst1dDesc );
         

      std::cout << "src2Desc Lengths: " << ref_src2dDesc.GetLength(Number<0>{})  << "," << ref_src2dDesc.GetLength(Number<1>{}) << std::endl; 
      std::cout << "sizeof src2Desc: " << sizeof(ref_src2dDesc) << ", sizeof dst1dDesc: " << sizeof(ref_dst1dDesc) << std::endl; 

}; 




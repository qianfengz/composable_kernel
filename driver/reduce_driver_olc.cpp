#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "olc_device_dynamic_generic_reduction.hpp"
#include "olc_driver_common.hpp"
#include "olc_reduce_common.hpp"
#include "host_generic_reduction.hpp"

#include "conv_tunables.hpp"
#include "handle.hpp"
#include "hipCheck.hpp"

void check_reduce_dims(const std::vector<int> &invariantDims, const std::vector<int> &toReduceDims, const int totalDims)
{
   auto tmpDims = invariantDims; 

   if ( invariantDims.size() + toReduceDims.size() != totalDims )
	throw std::runtime_error("Invalid number of dimensions specified for being invariant/toReduce"); 

   for (const auto dim :  toReduceDims)
	tmpDims.push_back(dim); 

   for (const auto dim : tmpDims) {
	if ( dim < 0 || dim >= tmpDims.size() ) 
	     throw std::runtime_error("Invalid dimension index specified for being invariant/toReduce"); 
   };

   for (int i1=0; i1 < tmpDims.size(); i1++) 
	for (int i2=0; i2 < tmpDims.size(); i2++) { 
             if (i1 != i2 && tmpDims[i1] == tmpDims[i2] ) 
	         throw std::runtime_error("Repeated dimension indexes specified for being invariant/toReduce"); 
	}; 
}; 

int main(int argc, char* argv[])
{
    using namespace ck;
    using size_t = std::size_t;

    hipStream_t stream;
    olCompile::Handle* handle;

    MY_HIP_CHECK(hipStreamCreate(&stream));

    handle = new olCompile::Handle(stream);

    const bool do_verification    = atoi(argv[1]);
    const int init_method         = atoi(argv[2]);
    const bool do_log             = atoi(argv[3]);
    const int nrepeat             = atoi(argv[4]);

    using srcDataType   = float;
    using dstDataType   = float;
    const appDataType_t compTypeId   = appFloat;

    using compType = Driver::get_type_from_type_enum<compTypeId>::type;

    float alpha = 1.0f; 
    float beta = 0.0f; 
    ReduceTensorOp_t reduceOp = ReduceTensorOp_t::REDUCE_TENSOR_ADD; 
    NanPropagation_t nanPropaOpt = NanPropagation_t::NOT_PROPAGATE_NAN;
    ReduceTensorIndices_t reduceIndiceOpt = ReduceTensorIndices_t::REDUCE_TENSOR_NO_INDICES; 

    std::vector<size_t> inLengths = {64L, 3L, 280L, 81L}; 
    std::vector<int> invariantDims = {1, 2, 3};  
    std::vector<int> toReduceDims = {0}; 

    std::vector<size_t> outLengths; 

    check_reduce_dims(invariantDims, toReduceDims, inLengths.size()); 

    if ( invariantDims.empty() ) {
	 outLengths.push_back(1); 
    }
    else {
         for (auto dim : invariantDims) 
	      outLengths.push_back(inLengths[dim]); 
    }; 
   
    Tensor<srcDataType> in(inLengths);
    Tensor<dstDataType> out_host(outLengths);
    Tensor<dstDataType> out_device(outLengths);
    Tensor<int> indices_host(outLengths); 

    ostream_HostTensorDescriptor(in.mDesc, std::cout << "in: ");
    ostream_HostTensorDescriptor(out_host.mDesc, std::cout << "out: ");

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            in.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_1{}, num_thread);
            break;
        case 2:
            in.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            break;
        case 3:
            in.GenerateTensorValue(GeneratorTensor_2{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_2{1, 5}, num_thread);
        }
    }

    tunable_dyn_generic_reduction* tunable = &default_tunable_dyn_generic_reduction;


    device_dynamic_generic_reduction_olc<srcDataType, compType, dstDataType>(handle, invariantDims, toReduceDims, in, out_device, reduceOp, nanPropaOpt, reduceIndiceOpt, alpha,  beta,  tunable, nrepeat);

    
    if(do_verification)
    {
        ReductionHost<srcDataType, dstDataType> hostReduce(reduceOp,  compTypeId, nanPropaOpt, reduceIndiceOpt, in.mDesc, out_host.mDesc, invariantDims, toReduceDims);  

        hostReduce.Run(alpha, in.mData.data(), beta, out_host.mData.data(), indices_host.mData.data()); 

        check_error(out_host, out_device);

        if(do_log)
        {
            LogRange(std::cout << "in : ", in.mData, ",") << std::endl;
            LogRange(std::cout << "out_host  : ", out_host.mData, ",") << std::endl;
            LogRange(std::cout << "out_device: ", out_device.mData, ",") << std::endl;
        }
    }

    delete handle;
    MY_HIP_CHECK(hipStreamDestroy(stream));
}

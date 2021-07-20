#include "device.hpp"
#include "host_tensor.hpp"

#include "olc_driver_common.hpp"
#include "olc_reduce_common.hpp"
#include "conv_tunables.hpp"

#include "handle.hpp"

namespace detail_dyn_generic_reduction {

template <typename TSrc, typename TComp, typename TDst>
static std::string get_network_config_string_from_types()
{
    std::string out;

    out += static_cast<char>(Driver::get_typeid_from_type<TSrc>()) +
           static_cast<char>(Driver::get_typeid_from_type<TComp>()) +
           static_cast<char>(Driver::get_typeid_from_type<TDst>());

    return (out);
};

static std::string
get_network_config_string_from_tunable(const tunable_dyn_generic_reduction* pt)
{
    std::string out("TUN_");

    out += std::to_string(pt->BlockSize) + "_";
    out += std::to_string(pt->GredThreadBufferLength) + "_";
    out += std::to_string(pt->GredAccessesPerThreadInBlock) + "_";
    out += std::to_string(pt->GredAccessesPerThreadInWarp); 

    return (out);
};

template <typename TSrc, typename TComp, typename TDst>
static std::string get_definition_string_from_types()
{
    std::string out;

    out += " -DCK_PARAM_SRC_DATATYPE=" + std::to_string(Driver::get_typeid_from_type<TSrc>());
    out += " -DCK_PARAM_DST_DATATYPE=" + std::to_string(Driver::get_typeid_from_type<TDst>());
    out += " -DCK_PARAM_REDUCE_COMPTYPE=" + std::to_string(Driver::get_typeid_from_type<TComp>());

    return (out);
};

static std::string
get_definition_string_from_tunable(const tunable_dyn_generic_reduction* pt)
{
    std::string out;

    out += " -DCK_PARAM_BLOCKSIZE=" + std::to_string(pt->BlockSize);
    out += " -DCK_PARAM_THREAD_BUFFER_LENGTH=" + std::to_string(pt->GredThreadBufferLength);
    out += " -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=" + std::to_string(pt->GredAccessesPerThreadInBlock);
    out += " -DCK_PARAM_ACCESSES_PER_THREAD_INWARP=" + std::to_string(pt->GredAccessesPerThreadInWarp);

    return (out);
};

struct ReductionKernelConfigurator
{
    ReductionKernelConfigurator() = default;

    ReductionKernelConfigurator(int blockSize, int warpSize)
        : blockSize_(blockSize), warpSize_(warpSize)
    {
        GredDirectThreadWiseUpperReductionLen = warpSize;
        GredDirectWarpWiseUpperReductionLen   = blockSize;
        GredBlockWiseUpperReductionLen        = blockSize * 4;
        GredUpperNumBlocksPerReduction        = 32;

        numWarpsPerBlock = blockSize / warpSize;
    };

    int blockSize_;
    int warpSize_;
    int numWarpsPerBlock;

    std::size_t GredDirectThreadWiseUpperReductionLen;
    std::size_t GredDirectWarpWiseUpperReductionLen;
    std::size_t GredBlockWiseUpperReductionLen;
    std::size_t GredUpperNumBlocksPerReduction;

    std::size_t getGridSize(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(invariantLength == 1)
        {
            if(toReduceLength <=
               GredBlockWiseUpperReductionLen) // let one block to do this only reduction
                return (1);
            else
                return ((toReduceLength + blockSize_ - 1) /
                        blockSize_); // let multiple blocks to do this only reduction
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
                return ((invariantLength + blockSize_ - 1) / blockSize_);
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
                return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
                return (invariantLength);
            else
            { // let multiple blocks to do each reduction
                std::size_t expBlocksPerReduction =
                    (toReduceLength + GredBlockWiseUpperReductionLen - 1) /
                    GredBlockWiseUpperReductionLen;

                if(expBlocksPerReduction > GredUpperNumBlocksPerReduction)
                    return (invariantLength * GredUpperNumBlocksPerReduction);
                else
                    return (invariantLength * expBlocksPerReduction);
            };
        };
    };

    ReductionMethod_t getReductionMethod(std::size_t invariantLength,
                                         std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(invariantLength == 1)
        {
            if(toReduceLength <=
               GredBlockWiseUpperReductionLen) // let one block to do this only reduction
                return (ReductionMethod_t::BlockWise);
            else // let multiple blocks to do this only reduction
                return (ReductionMethod_t::MultiBlock);
        }
        else
        {
            if(toReduceLength <=
               GredDirectThreadWiseUpperReductionLen) // let one thread to do each reduction
                return (ReductionMethod_t::DirectThreadWise);
            else if(toReduceLength <=
                    GredDirectWarpWiseUpperReductionLen) // let one warp to do each reduction
                return (ReductionMethod_t::DirectWarpWise);
            else if(toReduceLength <=
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
                return (ReductionMethod_t::BlockWise);
            else
                return (ReductionMethod_t::MultiBlock); // let multiple blocks to do each reduction
        };
    };

    std::size_t getWorkspaceSize(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        assert(invariantLength > 0 && toReduceLength > 1);

        if(getReductionMethod(invariantLength, toReduceLength) == ReductionMethod_t::MultiBlock)
        {
            auto gridSize = getGridSize(invariantLength, toReduceLength);

            return (gridSize);
        };

        return (0);
    };

    std::size_t getGridSize_2(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        if(toReduceLength <= warpSize_ / 4) // let one thread to do each reduction
            return ((invariantLength + blockSize_ - 1) / blockSize_);
        else if(toReduceLength <= blockSize_) // let one warp to do each reduction
            return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
        else
            return (invariantLength); // let one block to do each reduction
    };

    ReductionMethod_t GetReductionMethod_2(std::size_t invariantLength, std::size_t toReduceLength) const
    {
        if(toReduceLength <= warpSize_ / 4) // let one thread to do each reduction
            return (ReductionMethod_t::DirectThreadWise);
        else if(toReduceLength <= blockSize_) // let one warp to do each reduction
            return (ReductionMethod_t::DirectWarpWise);
        else
            return (ReductionMethod_t::BlockWise);
    };
};

static inline int GetDataTypeId(appDataType_t t)
{
    switch(t)
    {
    case appHalf: return (static_cast<int>('H'));
    case appFloat: return (static_cast<int>('F'));
    case appBFloat16: return (static_cast<int>('B'));
    case appDouble: return (static_cast<int>('D'));
    case appInt8:
    case appInt8x4:
    case appInt32: return (static_cast<int>('O'));
    default: 
	 throw std::runtime_error("Only float, half, bfloat16 data type is supported."); 
	 break;
    };
};

static inline int GetReduceTensorOpId(ReduceTensorOp_t t)
{
    switch(t)
    {
    case REDUCE_TENSOR_ADD:
        return (656868); // 'A' * 10000 + 'D' * 100 + 'D'
    case REDUCE_TENSOR_MUL:
        return (778576); // 'M' * 10000 + 'U' * 100 + 'L'
    case REDUCE_TENSOR_MIN:
        return (777378); // 'M' * 10000 + 'I' * 100 + 'N'
    case REDUCE_TENSOR_MAX:
        return (776588); // 'M' * 10000 + 'A' * 100 + 'X'
    case REDUCE_TENSOR_AMAX:
        return (657788); // 'A' * 10000 + 'M' * 100 + 'X'
    case REDUCE_TENSOR_AVG:
        return (658671); // 'A' * 10000 + 'V' * 100 + 'G'
    case REDUCE_TENSOR_NORM1:
        return (788201); // 'N' * 10000 + 'R' * 100 + '1'
    case REDUCE_TENSOR_NORM2:
        return (788202); // 'N' * 10000 + 'R' * 100 + '2'
    default:
        throw std::runtime_error("Operation is not supported"); 
 	break;
    };
};

} // namespace detail_dyn_generic_reduction

template <typename TSrc,
          typename TComp,
          typename TDst>
void device_dynamic_generic_reduction_olc(
    olCompile::Handle* handle,
    const std::vector<int> invariantDims,
    const std::vector<int> toReduceDims,
    const Tensor<TSrc>& in,
    Tensor<TDst>& out,
    ReduceTensorOp_t reduceOp, 
    NanPropagation_t nanPropaOpt,
    ReduceTensorIndices_t reduceIndicesOpt, 
    float alpha, 
    float beta,
    const tunable_dyn_generic_reduction* tunable,
    ck::index_t nrepeat)
{
    using namespace ck;
    using namespace detail_dyn_generic_reduction;
    using size_t = std::size_t;


    size_t invariantLength = out.mDesc.GetElementSize();
    size_t toReduceLength = in.mDesc.GetElementSize() / invariantLength;
    int origReduceLen = toReduceLength;

    ReductionKernelConfigurator configurator(tunable->BlockSize, handle->GetWavefrontWidth());

    // these buffers are usually provided by the user application
    DeviceMem in_dev_buf(sizeof(TSrc) * in.mDesc.GetElementSpace());
    DeviceMem out_dev_buf(sizeof(TDst) * out.mDesc.GetElementSpace());

    in_dev_buf.ToDevice(in.mData.data()); 
   
    if ( beta != 0.0f )
	 out_dev_buf.ToDevice(out.mData.data());  

    auto inLengths = in.mDesc.GetLengths(); 
    auto inStrides = in.mDesc.GetStrides(); 
    auto outLengths = out.mDesc.GetLengths(); 
    auto outStrides = out.mDesc.GetStrides(); 

    std::vector<index_t> lens_buf(4096/sizeof(index_t));   // allocate one page  

    for (int i=0; i < inLengths.size(); i++) 
         lens_buf[0+i] = static_cast<index_t>(inLengths[i]); 

    for (int i=0; i < inStrides.size(); i++) 
         lens_buf[6+i] = static_cast<index_t>(inStrides[i]); 

    for (int i=0; i < outLengths.size(); i++) 
         lens_buf[12+i] = static_cast<index_t>(outLengths[i]); 

    for (int i=0; i < outStrides.size(); i++)
	 lens_buf[18+i] = static_cast<index_t>(outStrides[i]); 

    auto workspace_size = configurator.getWorkspaceSize(invariantLength, toReduceLength);

    bool need_indices = (reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES) &&
        (reduceOp == REDUCE_TENSOR_MIN || reduceOp == REDUCE_TENSOR_MAX || reduceOp == REDUCE_TENSOR_AMAX);

    size_t wsSizeInBytes = 
        !need_indices ? workspace_size * sizeof(TSrc) 
                      : workspace_size * (sizeof(TSrc) + sizeof(int)) + 64 + sizeof(int);

    DeviceMem workspace1(wsSizeInBytes); 
    DeviceMem workspace2(4096); 

    void *p_dev_src2dDesc = (char *)workspace2.GetDeviceBuffer() + 1024; 
    void *p_dev_dst1dDesc = (char *)workspace2.GetDeviceBuffer() + 2048;
    bool *p_dev_src_use_padding = reinterpret_cast<bool *>( (char *)workspace2.GetDeviceBuffer() + 3072 ); 
    bool *p_dev_dst_use_padding = reinterpret_cast<bool *>( (char *)workspace2.GetDeviceBuffer() + 3072 + sizeof(size_t) ); 
    index_t *p_dev_inLengths = (index_t *)workspace2.GetDeviceBuffer(); 
    index_t *p_dev_inStrides = &p_dev_inLengths[6]; 
    index_t *p_dev_outLengths = &p_dev_inLengths[12]; 
    index_t *p_dev_outStrides = &p_dev_inLengths[18]; 

    workspace2.ToDevice(static_cast<const void*>(lens_buf.data()));  

    size_t indicesSizeInBytes = need_indices? out.mDesc.GetElementSize() * sizeof(int) : 0; 

    DeviceMem indices_dev_buf(indicesSizeInBytes);

    size_t ws_buf2_bytes_offset = 0; 

    if(need_indices && workspace_size > 0)
    {
        size_t byteOffset =
            static_cast<size_t>((wsSizeInBytes / (sizeof(TSrc) + sizeof(int))) * sizeof(TSrc));

        ws_buf2_bytes_offset = ((byteOffset + 63) / 64) * 64;
    };

    ReductionMethod_t reduceImpl = configurator.getReductionMethod(invariantLength, toReduceLength);
    int GridSize                = static_cast<int>( configurator.getGridSize(invariantLength, toReduceLength) ); 
    int BlkGroupSize = (reduceImpl == ReductionMethod_t::MultiBlock) ? GridSize / invariantLength : 0;    

    const std::vector<size_t> vld  = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd1 = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd2 = {static_cast<size_t>(GridSize) * tunable->BlockSize, 1, 1};

    std::string program_name = "dynamic_gridwise_generic_reduction.cpp";
    std::string algo_name    = "dynamic_generic_reduction";

    std::string param = " -std=c++17 ";
    std::string network_config;

    param += get_definition_string_from_types<TSrc, TComp, TDst>() + " " + get_definition_string_from_tunable(tunable);

    param += " -DCK_PARAM_TOREDUCE_DIMS=";
    for(int i = 0; i < toReduceDims.size(); i++)
    {
        param += std::to_string(toReduceDims[i]);
        if(i < toReduceDims.size() - 1)
            param += ",";
    };

    if(!invariantDims.empty())
    {
        param += " -DCK_PARAM_INVARIANT_DIMS=";
        for(int i = 0; i < invariantDims.size(); i++)
        {
            param += std::to_string(invariantDims[i]);
            if(i < invariantDims.size() - 1)
                param += ",";
        };
    }
    else {
        param += " -DCK_PARAM_INVARIANT_DIMS= ";
        param += " -DCK_REDUCE_ALL_DIMS=1"; 
    }; 

    param += " -DCK_PARAM_REDUCE_OP=" + std::to_string(GetReduceTensorOpId(reduceOp));
    param += " -DCK_PARAM_NAN_PROPAGATE=" + std::to_string(nanPropaOpt == PROPAGATE_NAN ? 1 : 0);
    param += " -DCK_PARAM_REDUCE_INDICES=" + std::to_string(reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES ? 1 : 0);
    param += " -DCK_PARAM_IN_DIMS=" + std::to_string(inLengths.size()); 
    param += " -DCK_PARAM_OUT_DIMS=" + std::to_string(outLengths.size()); 
	     
    network_config = get_network_config_string_from_types<TSrc, TComp, TDst>() + "_" + get_network_config_string_from_tunable(tunable) + "_"; 

    network_config += "I" + std::to_string(inLengths.size()) + "_"; 

    network_config += "RED";
    for(auto dim : toReduceDims)
        network_config += std::to_string(dim) + "_";
    network_config += "BSIZE_" + std::to_string(tunable->BlockSize);

    std::cout << std::endl << "Reduction method=" << reduceImpl << " GridSize=" << GridSize << " BlkGroupSize=" << BlkGroupSize << std::endl;

    std::vector<float> kernel1_times;
    std::vector<float> kernel2_times;
    std::vector<float> kernel3_times;
    std::vector<float> kernel4_times;

    for(index_t i = 0; i < nrepeat; ++i)
    {
        KernelTimer timer1, timer2;
        std::string kernel_name;

        kernel_name = "gridwise_generic_reduce_1_prepare";
        auto network_config_1 = network_config + "_1";

        timer1.Start();
        handle->AddKernel(algo_name, network_config_1, program_name, kernel_name, vld, vgd1, param)(static_cast<int>(reduceImpl), GridSize, BlkGroupSize, p_dev_inLengths, p_dev_inStrides, p_dev_outLengths, p_dev_outStrides, 
			                                                                                          p_dev_src2dDesc, p_dev_dst1dDesc, p_dev_src_use_padding, p_dev_dst_use_padding); 
        timer1.End();

        kernel_name           = "gridwise_generic_reduce_1";
        auto network_config_2 = network_config + "_2";

        timer2.Start();
        handle->AddKernel(algo_name, network_config_2, program_name, kernel_name, vld, vgd2, param)(static_cast<int>(reduceImpl), origReduceLen, BlkGroupSize, p_dev_src2dDesc, p_dev_dst1dDesc, p_dev_src_use_padding, p_dev_dst_use_padding, 
		                                                            alpha, in_dev_buf.GetDeviceBuffer(), beta, out_dev_buf.GetDeviceBuffer(), workspace1.GetDeviceBuffer(), ws_buf2_bytes_offset, indices_dev_buf);  
        timer2.End();

        kernel1_times.push_back(timer1.GetElapsedTime());
        kernel2_times.push_back(timer2.GetElapsedTime());

        if (reduceImpl == ReductionMethod_t::MultiBlock) {
            auto toReduceLength_2 = BlkGroupSize;
            auto GridSize_2       = configurator.getGridSize_2(invariantLength, toReduceLength_2);
            const std::vector<size_t> vgd2_2 = {GridSize_2 * tunable->BlockSize, size_t{1}, size_t{1}};
            auto reduceImpl2 = configurator.GetReductionMethod_2(invariantLength, toReduceLength_2); 

            kernel_name = "gridwise_generic_reduce_2_prepare";
            network_config_1 = network_config + "_1";

            timer1.Start(); 
            handle->AddKernel(algo_name, network_config_1, program_name, kernel_name, vld, vgd1, param)(static_cast<int>(reduceImpl2), GridSize_2, BlkGroupSize, p_dev_inLengths, p_dev_inStrides, p_dev_outLengths, p_dev_outStrides,
                                                                                                                  p_dev_src2dDesc, p_dev_dst1dDesc, p_dev_src_use_padding, p_dev_dst_use_padding);
	    timer1.End(); 

            kernel_name = "gridwise_generic_reduce_2";
            network_config_2 = network_config + "_2";

            timer2.Start(); 
            handle->AddKernel(algo_name, network_config_2, program_name, kernel_name, vld, vgd2_2, param)(static_cast<int>(reduceImpl2), origReduceLen, BlkGroupSize, p_dev_src2dDesc, p_dev_dst1dDesc, p_dev_src_use_padding, p_dev_dst_use_padding,
		                                                            alpha, in_dev_buf.GetDeviceBuffer(), beta, out_dev_buf.GetDeviceBuffer(), workspace1.GetDeviceBuffer(), ws_buf2_bytes_offset, indices_dev_buf);  
	    timer1.End(); 

            kernel3_times.push_back(timer1.GetElapsedTime());
            kernel4_times.push_back(timer2.GetElapsedTime());
	}; 
    }

    {
        auto ave_time1 = Driver::get_effective_average(kernel1_times);
        auto ave_time2 = Driver::get_effective_average(kernel2_times);

        if ( reduceImpl == ReductionMethod_t::MultiBlock ) {
             auto ave_time3 = Driver::get_effective_average(kernel3_times);
             auto ave_time4 = Driver::get_effective_average(kernel4_times);

             std::cout << "Average time : " << ave_time1 + ave_time2 + ave_time3 + ave_time3 << " ms(" << ave_time1+ave_time3 << ", " << ave_time2+ave_time4 << ")" << std::endl;
        }
	else {
             std::cout << "Average time : " << ave_time1 + ave_time2 << " ms(" << ave_time1 << ", " << ave_time2 << ")" << std::endl;
        }; 
    };

    // copy result back to host
    out_dev_buf.FromDevice(out.mData.data());
}

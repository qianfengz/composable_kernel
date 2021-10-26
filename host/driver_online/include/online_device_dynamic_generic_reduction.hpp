#include "device.hpp"
#include "host_tensor.hpp"

#include "online_reduce_common.hpp"
#include "reduce_tunables.hpp"

#include "handle.hpp"

#include <sstream>
#include <cstdlib>

// headers from composable kernel, to get consistent ID mapping
#include <data_type_enum.hpp>
#include <reduction_enums.hpp>

static constexpr int default_workgroup_size = 256;

namespace detail_dyn_generic_reduction {

static int getEnvVarValue(const char* envVarName)
{
    const char* valString = getenv(envVarName);

    if(valString == nullptr)
        return (-1);

    int val = atoi(valString);

    if(val < 0)
        return (-1);

    return (val);
};

static ck::DataTypeEnum_t mapDataTypeId(appDataType_t t)
{
    using ck::DataTypeEnum_t;

    switch(t)
    {
    case appHalf: return DataTypeEnum_t::Half;
    case appFloat: return DataTypeEnum_t::Float;
    case appBFloat16: return DataTypeEnum_t::BFloat16;
    case appDouble: return DataTypeEnum_t::Double;
    case appInt8: return DataTypeEnum_t::Int8;
    case appInt8x4: return DataTypeEnum_t::Int8x4;
    case appInt32: return DataTypeEnum_t::Int32;
    default: throw std::runtime_error("Only float, half, double data type is supported.");
    };
};

static ck::ReduceTensorOp_t mapReduceOpId(ReduceTensorOp_t t)
{
    switch(t)
    {
    case REDUCE_TENSOR_ADD: return ck::ReduceTensorOp_t::ADD;
    case REDUCE_TENSOR_MUL: return ck::ReduceTensorOp_t::MUL;
    case REDUCE_TENSOR_MIN: return ck::ReduceTensorOp_t::MIN;
    case REDUCE_TENSOR_MAX: return ck::ReduceTensorOp_t::MAX;
    case REDUCE_TENSOR_AMAX: return ck::ReduceTensorOp_t::AMAX;
    case REDUCE_TENSOR_AVG: return ck::ReduceTensorOp_t::AVG;
    case REDUCE_TENSOR_NORM1: return ck::ReduceTensorOp_t::NORM1;
    case REDUCE_TENSOR_NORM2: return ck::ReduceTensorOp_t::NORM2;

    default: throw std::runtime_error("Operation is not supported");
    };
};

template <typename TSrc, typename TComp, typename TDst>
static std::string get_network_config_string_from_types()
{
    std::ostringstream outs;

    outs << Driver::get_type_enum_from_type<TSrc>() << Driver::get_type_enum_from_type<TComp>()
         << Driver::get_type_enum_from_type<TDst>();

    return (outs.str());
};

template <typename TSrc, typename TComp, typename TDst>
static std::string get_definition_string_from_types()
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_SRC_DATATYPE=" << mapDataTypeId(Driver::get_type_enum_from_type<TSrc>());
    outs << " -DCK_PARAM_DST_DATATYPE=" << mapDataTypeId(Driver::get_type_enum_from_type<TDst>());
    outs << " -DCK_PARAM_REDUCE_COMPTYPE="
         << mapDataTypeId(Driver::get_type_enum_from_type<TComp>());

    return (outs.str());
};

static tunable_generic_2d_reduction getDefaultTunable(ReductionMethod_t reduceImpl)
{
    tunable_generic_2d_reduction tunable;

    tunable.BlockSize                  = default_workgroup_size;
    tunable.dim0_thread_cluster_length = 1;
    tunable.dim0_thread_slice_length   = 1;
    tunable.dim1_thread_cluster_length = default_workgroup_size;

    if(reduceImpl == ReductionMethod_t::DirectThreadWise)
        tunable.dim1_thread_slice_length = 8;
    else
        tunable.dim1_thread_slice_length = 2;
    tunable.reordered_thread_clusters = false;

    return (tunable);
};

static std::string get_basic_network_config_string(ReductionMethod_t reduceImpl,
                                                   bool useGlobalAtomicAdd = false)
{
    std::ostringstream outs;

    outs << "NC_" << reduceImpl << "_" << useGlobalAtomicAdd;

    return (outs.str());
};

static std::string get_network_config_string_from_tunable(ReductionMethod_t reduceImpl,
                                                          const tunable_generic_2d_reduction* pt,
                                                          bool useGlobalAtomicAdd = false)
{
    std::ostringstream outs;

    outs << "TUN_" << pt->BlockSize << "_";

    if(reduceImpl == ReductionMethod_t::MultiBlock && useGlobalAtomicAdd)
    {
#ifdef TEST_GENERIC_CONFIG
        outs << pt->dim0_thread_cluster_length << "_" << pt->dim0_thread_slice_length << "_";
        outs << pt->dim1_thread_cluster_length << "_" << pt->dim1_thread_slice_length << "_";
        outs << pt->reordered_thread_clusters;
#else
        outs << pt->dim1_thread_slice_length << "_";
#endif
    }
    else
        outs << pt->dim1_thread_slice_length << "_";

    return (outs.str());
};

static std::string get_definition_string_from_tunable(ReductionMethod_t reduceImpl,
                                                      const tunable_generic_2d_reduction* pt,
                                                      bool useGlobalAtomicAdd = false)
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_BLOCKSIZE=" << pt->BlockSize;

    if(reduceImpl == ReductionMethod_t::DirectThreadWise)
        outs << " -DCK_PARAM_THREAD_BUFFER_LENGTH=" << pt->dim1_thread_slice_length;

    if(reduceImpl == ReductionMethod_t::DirectWarpWise)
        outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INWARP=" << pt->dim1_thread_slice_length;

    if(reduceImpl == ReductionMethod_t::BlockWise)
        outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=" << pt->dim1_thread_slice_length;

    if(reduceImpl == ReductionMethod_t::MultiBlock)
    {
        if(useGlobalAtomicAdd)
        {
#ifdef TEST_GENERIC_CONFIG
            outs << " -DCK_PARAM_DIM0_THREAD_CLUSTER_LENGTH=" << pt->dim0_thread_cluster_length;
            outs << " -DCK_PARAM_DIM0_THREAD_SLICE_LENGTH=" << pt->dim0_thread_slice_length;
            outs << " -DCK_PARAM_DIM1_THREAD_CLUSTER_LENGTH=" << pt->dim1_thread_cluster_length;
            outs << " -DCK_PARAM_DIM1_THREAD_SLICE_LENGTH=" << pt->dim1_thread_slice_length;
            outs << " -DCK_PARAM_REORDERED_THREAD_CLUSTERS=" << pt->reordered_thread_clusters;
#else
            outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=" << pt->dim1_thread_slice_length;
#endif
        }
        else
            outs << " -DCK_PARAM_ACCESSES_PER_THREAD_INBLOCK=" << pt->dim1_thread_slice_length;
    }

    return (outs.str());
};

static int getBlocksForEnoughUtilization(const online_compile::Handle* handle, int BlockSize)
{
    int numCUs           = handle->GetMaxComputeUnits();
    int numWarpsPerBlock = BlockSize / handle->GetWavefrontWidth();

    // assumes: 1) 4 SIMDs per CU and assume
    //          2) need 30 active waves for a complete utilization of one SIMD
    return (numCUs * 4 * 30 / numWarpsPerBlock);
};

struct ReductionKernelConfigurator
{
    ReductionKernelConfigurator() = default;

    ReductionKernelConfigurator(int blockSize, int warpSize, int numMaxCUs)
        : blockSize_(blockSize), warpSize_(warpSize), numMaxCUs_(numMaxCUs)
    {
        GredDirectThreadWiseUpperReductionLen = warpSize;
        GredDirectWarpWiseUpperReductionLen   = blockSize;
        GredBlockWiseUpperReductionLen        = blockSize * 4;
        GredUpperNumBlocksPerReduction        = 128;

        numWarpsPerBlock = blockSize / warpSize;
    };

    int blockSize_;
    int warpSize_;
    int numWarpsPerBlock;
    int numMaxCUs_;

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
            {
                // let multiple blocks to do each reduction
                std::size_t expBlocksPerReduction =
                    (toReduceLength + GredBlockWiseUpperReductionLen - 1) /
                    GredBlockWiseUpperReductionLen;

                if(expBlocksPerReduction > GredUpperNumBlocksPerReduction)
                {
                    if(invariantLength >= numMaxCUs_)
                    {
                        if(expBlocksPerReduction < 2 * GredUpperNumBlocksPerReduction)
                            return (invariantLength * expBlocksPerReduction);
                        else
                            return (invariantLength * GredUpperNumBlocksPerReduction);
                    }
                    else
                    {
                        int numBlocksPerReduce = GredUpperNumBlocksPerReduction;

                        while(numBlocksPerReduce <= expBlocksPerReduction)
                        {
                            // increase the number of blocks per reduction so that we have enough
                            // blocks to utlize the CUs
                            if(invariantLength * numBlocksPerReduce >= numMaxCUs_ * 4)
                                return (invariantLength * numBlocksPerReduce);
                            numBlocksPerReduce++;
                        };
                        return (invariantLength * expBlocksPerReduction);
                    }
                }
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
            {
                return (ReductionMethod_t::MultiBlock); // let multiple blocks to do each reduction
            }
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

    ReductionMethod_t GetReductionMethod_2(std::size_t invariantLength,
                                           std::size_t toReduceLength) const
    {
        if(toReduceLength <= warpSize_ / 4) // let one thread to do each reduction
            return (ReductionMethod_t::DirectThreadWise);
        else if(toReduceLength <= blockSize_) // let one warp to do each reduction
            return (ReductionMethod_t::DirectWarpWise);
        else
            return (ReductionMethod_t::BlockWise);
    };
};

static std::pair<bool, bool> get_padding_need(ReductionMethod_t reduceImpl,
                                              size_t invariantLen,
                                              size_t toReduceLen,
                                              int GridSize,
                                              int BlockSize,
                                              int warpSize,
                                              int BlkGroupSize,
                                              const tunable_generic_2d_reduction* tunable)
{
    bool src_need_padding = false;
    bool dst_need_padding = false;
    int copySliceLen;
    int reduceSizePerBlock;

    switch(reduceImpl)
    {
    case ReductionMethod_t::DirectThreadWise:
        copySliceLen = tunable->dim1_thread_slice_length;
        src_need_padding =
            (invariantLen < GridSize * BlockSize || toReduceLen % copySliceLen > 0) ? true : false;
        dst_need_padding = (invariantLen < GridSize * BlockSize) ? true : false;
        break;
    case ReductionMethod_t::DirectWarpWise:
        copySliceLen = warpSize * tunable->dim1_thread_slice_length;
        src_need_padding =
            (invariantLen < GridSize * BlockSize / warpSize || toReduceLen % copySliceLen > 0)
                ? true
                : false;
        dst_need_padding = (invariantLen < GridSize * BlockSize / warpSize) ? true : false;
        break;
    case ReductionMethod_t::BlockWise:
        copySliceLen     = BlockSize * tunable->dim1_thread_slice_length;
        src_need_padding = (toReduceLen % copySliceLen > 0) ? true : false;
        break;
    case ReductionMethod_t::MultiBlock:
        copySliceLen = BlockSize * tunable->dim1_thread_slice_length;
        reduceSizePerBlock =
            (((toReduceLen + BlkGroupSize - 1) / BlkGroupSize + copySliceLen - 1) / copySliceLen) *
            copySliceLen;
        src_need_padding = (toReduceLen < reduceSizePerBlock * BlkGroupSize) ? true : false;
        break;
    default: throw std::runtime_error("Invalid reduction method ID!"); break;
    };

    return (std::make_pair(src_need_padding, dst_need_padding));
};

static std::string get_network_config_string_from_options(NanPropagation_t nanPropaOpt,
                                                          ReduceTensorIndices_t reduceIndicesOpt)
{
    std::ostringstream outs;

    outs << "O_" << ((nanPropaOpt == PROPAGATE_NAN) ? 1 : 0)
         << ((reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES) ? 1 : 0);

    return (outs.str());
};

static std::string get_definition_string_from_options(NanPropagation_t nanPropaOpt,
                                                      ReduceTensorIndices_t reduceIndicesOpt)
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_NAN_PROPAGATE=" << ((nanPropaOpt == PROPAGATE_NAN) ? 1 : 0);
    outs << " -DCK_PARAM_REDUCE_INDICES="
         << ((reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES) ? 1 : 0);

    return (outs.str());
};

static std::string getReductionMethodStr(ReductionMethod_t reduceImpl)
{
    switch(reduceImpl)
    {
    case ReductionMethod_t::DirectThreadWise: return (std::string("threadwise"));
    case ReductionMethod_t::DirectWarpWise: return (std::string("warpwise"));
    case ReductionMethod_t::BlockWise: return (std::string("blockwise"));
    case ReductionMethod_t::MultiBlock: return (std::string("multiblock"));
    default: throw std::runtime_error("Invalid reduction method ID!"); break;
    };
};

static inline std::string get_arch_specific_compiler_flag(const online_compile::Handle* handle)
{
    std::string compiler_flag;

    // GPU target
    static const std::string gpu_target = handle->GetDeviceName();

    if(gpu_target.compare(0, 6, "gfx803") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX803";
    else if(gpu_target.compare(0, 6, "gfx900") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX900";
    else if(gpu_target.compare(0, 6, "gfx906") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX906";
    else if(gpu_target.compare(0, 6, "gfx908") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX908";
    else if(gpu_target.compare(0, 6, "gfx90a") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX90A";
    else if(gpu_target.compare(0, 7, "gfx1030") == 0)
        compiler_flag = " -DCK_AMD_GPU_GFX1030";

    return compiler_flag;
};

static bool useGlobalAtomicAddReduce(const ReduceTensorOp_t reduceOp,
                                     appDataType_t globalOutDataType,
                                     float beta)
{
    if(getEnvVarValue("DEBUG_REDUCE_USE_ATOMIC_ADD") == 0)
        return (false);

    if(beta != 0.0f)
        return (false);

    // AtomicAdd for Half is not supported by the Hardware, while for Double
    // change in amd_buffer_addressing.hpp is needed to support AtomicAdd
    if(globalOutDataType != appFloat)
        return (false);

    if(reduceOp != REDUCE_TENSOR_ADD && reduceOp != REDUCE_TENSOR_AVG &&
       reduceOp != REDUCE_TENSOR_NORM1)
        return (false);

    return (true);
};

static bool useVectorLoadOnInvariantDim(const ReductionMethod_t reduceImpl,
                                        int InvariantDimVectorSize,
                                        bool use_indices)
{
    if(getEnvVarValue("DEBUG_USE_INVARIANT_DIM_VECTOR_LOAD") == 0)
        return (false);

    if(use_indices)
        return (false);

    // At first, vector_load on invariant dim can only be used with MultiBlock kernel, gradually
    // it will be extended for using with other reduction kernel
    if(reduceImpl != ReductionMethod_t::MultiBlock)
        return (false);

    if(InvariantDimVectorSize <= 1)
        return (false);

    return (true);
};

static std::string get_kernel_file_name(const bool isSecondCall,
                                        const ReductionMethod_t reduceImpl,
                                        const bool allDimsReduced,
                                        const bool useGlobalAtomicAdd = false)
{
    std::ostringstream outs;

    if(reduceImpl == ReductionMethod_t::MultiBlock && useGlobalAtomicAdd)
    {
        outs << "gridwise_generic_reduction_" << getReductionMethodStr(reduceImpl);

        if(allDimsReduced)
            outs << "_reduce_all_dims";
        else
            outs << "_reduce_partial_dims";

        outs << "_atomic_add.cpp";
    }
    else
    {
        if(!isSecondCall)
            outs << "gridwise_generic_reduction_" << getReductionMethodStr(reduceImpl);
        else
            outs << "gridwise_generic_reduction_second_call_" << getReductionMethodStr(reduceImpl);

        if(allDimsReduced)
            outs << "_reduce_all_dims.cpp";
        else
            outs << "_reduce_partial_dims.cpp";
    }

    return (outs.str());
};

static int merge_packed_dimensions(int* dimLengths, int* dimStrides, int numDims)
{
    assert(numDims >= 1);

    int resNumDims = numDims;
    int pos        = numDims - 1;

    while(pos > 0)
    {
        // packed dimensions
        if(dimStrides[pos - 1] == dimLengths[pos] * dimStrides[pos])
        {
            dimLengths[pos] = dimLengths[pos] * dimLengths[pos - 1];

            // shift the lower lengths/strides to left by 1
            for(int i = pos; i < resNumDims; i++)
            {
                dimLengths[i - 1] = dimLengths[i];
                dimStrides[i - 1] = dimStrides[i];
            };
            resNumDims--;
            pos--;
        }
        else
            pos--;
    };

    return (resNumDims);
};

template <typename dataType>
static int get_dim_vector_size(int dimLength, int dimStride)
{
    appDataType_t dataTypeId = Driver::get_type_enum_from_type<dataType>();

    if(dimStride != 1)
        return (1);

    if(dataTypeId != appDouble && dimLength % 8 == 0)
        return (8);

    if(dimLength % 4 == 0)
        return (4);

    if(dimLength % 2 == 0)
        return (2);

    return (1);
};

} // namespace detail_dyn_generic_reduction

template <typename TSrc, typename TComp, typename TDst>
void device_dynamic_generic_reduction_olc(online_compile::Handle* handle,
                                          const std::vector<int> invariantDims,
                                          const std::vector<int> toReduceDims,
                                          const Tensor<TSrc>& in,
                                          Tensor<TDst>& out,
                                          Tensor<int>& out_indices,
                                          ReduceTensorOp_t reduceOp,
                                          NanPropagation_t nanPropaOpt,
                                          ReduceTensorIndices_t reduceIndicesOpt,
                                          float alpha,
                                          float beta,
                                          int nrepeat)
{
    using namespace ck;
    using namespace detail_dyn_generic_reduction;
    using size_t = std::size_t;

    size_t invariantLength = out.mDesc.GetElementSize();
    size_t toReduceLength  = in.mDesc.GetElementSize() / invariantLength;
    int origReduceLen      = toReduceLength;

    ReductionKernelConfigurator configurator(
        default_workgroup_size, handle->GetWavefrontWidth(), handle->GetMaxComputeUnits());

    const bool reduceAllDims = invariantDims.empty();

    // these buffers are usually provided by the user application
    DeviceMem in_dev_buf(sizeof(TSrc) * in.mDesc.GetElementSpace());
    DeviceMem out_dev_buf(sizeof(TDst) * out.mDesc.GetElementSpace());

    in_dev_buf.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev_buf.ToDevice(out.mData.data());

    auto inLengths  = in.mDesc.GetLengths();
    auto inStrides  = in.mDesc.GetStrides();
    auto outLengths = out.mDesc.GetLengths();
    auto outStrides = out.mDesc.GetStrides();

    int p_inLengths[6]  = {0};
    int p_inStrides[6]  = {0};
    int p_outLengths[6] = {0};
    int p_outStrides[6] = {0};

    // re-order the input dimensions
    int pos = 0;

    for(int i = 0; i < invariantDims.size(); i++)
    {
        p_inLengths[pos] = static_cast<int>(inLengths[invariantDims[i]]);
        p_inStrides[pos] = static_cast<int>(inStrides[invariantDims[i]]);
        pos++;
    }

    for(int i = 0; i < toReduceDims.size(); i++)
    {
        p_inLengths[pos] = static_cast<int>(inLengths[toReduceDims[i]]);
        p_inStrides[pos] = static_cast<int>(inStrides[toReduceDims[i]]);
        pos++;
    }

    for(int i = 0; i < outLengths.size(); i++)
        p_outLengths[i] = static_cast<int>(outLengths[i]);

    for(int i = 0; i < outStrides.size(); i++)
        p_outStrides[i] = static_cast<int>(outStrides[i]);

    if(invariantDims.empty())
    {
        p_outLengths[0] = 1;
        p_outStrides[0] = 1;
    }

    int mergedInvariantDims =
        reduceAllDims ? 0 : merge_packed_dimensions(p_inLengths, p_inStrides, invariantDims.size());
    int mergedOutDims =
        reduceAllDims ? 1
                      : merge_packed_dimensions(p_outLengths, p_outStrides, invariantDims.size());

    int tmpPos = invariantDims.size();
    int mergedToReduceDims =
        merge_packed_dimensions(&p_inLengths[tmpPos], &p_inStrides[tmpPos], toReduceDims.size());

    // pack p_inLengths[] and p_inStrides[]
    if(invariantDims.size() > 0 && invariantDims.size() > mergedInvariantDims)
    {
        for(int i = 0; i < mergedToReduceDims; i++)
        {
            p_inLengths[mergedInvariantDims + i] = p_inLengths[tmpPos + i];
            p_inStrides[mergedInvariantDims + i] = p_inStrides[tmpPos + i];
        };
    };

    int ReduceDimVectorSize =
        get_dim_vector_size<TSrc>(p_inLengths[mergedInvariantDims + mergedToReduceDims - 1],
                                  p_inStrides[mergedInvariantDims + mergedToReduceDims - 1]);
    int InvariantDimVectorSize = get_dim_vector_size<TSrc>(p_inLengths[mergedInvariantDims - 1],
                                                           p_inStrides[mergedInvariantDims - 1]);

    auto workspace_size = configurator.getWorkspaceSize(invariantLength, toReduceLength);

    bool need_indices = (reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES) &&
                        (reduceOp == REDUCE_TENSOR_MIN || reduceOp == REDUCE_TENSOR_MAX ||
                         reduceOp == REDUCE_TENSOR_AMAX);

    size_t wsSizeInBytes = !need_indices
                               ? workspace_size * sizeof(TSrc)
                               : workspace_size * (sizeof(TSrc) + sizeof(int)) + 64 + sizeof(int);

    DeviceMem workspace(4096 + wsSizeInBytes);

    size_t indicesSizeInBytes = need_indices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

    DeviceMem indices_dev_buf(indicesSizeInBytes);

    size_t ws_buf2_bytes_offset = 0;

    if(need_indices && workspace_size > 0)
    {
        size_t byteOffset =
            static_cast<size_t>((wsSizeInBytes / (sizeof(TSrc) + sizeof(int))) * sizeof(TSrc));

        ws_buf2_bytes_offset = ((byteOffset + 63) / 64) * 64;
    };

    ReductionMethod_t reduceImpl = configurator.getReductionMethod(invariantLength, toReduceLength);
    int GridSize = static_cast<int>(configurator.getGridSize(invariantLength, toReduceLength));
    int BlkGroupSize =
        (reduceImpl == ReductionMethod_t::MultiBlock) ? GridSize / invariantLength : 0;

    const bool useGlobalAtomicAdd =
        useGlobalAtomicAddReduce(reduceOp, Driver::get_type_enum_from_type<TDst>(), beta);

    const bool InvariantDimVectorLoad =
        useVectorLoadOnInvariantDim(reduceImpl, InvariantDimVectorSize, need_indices);

    auto tunable = getDefaultTunable(reduceImpl);

    const std::vector<size_t> vld    = {static_cast<size_t>(tunable.BlockSize), 1, 1};
    const std::vector<size_t> vgd1   = {static_cast<size_t>(tunable.BlockSize), 1, 1};
    const std::vector<size_t> vgd1_s = {
        (invariantLength + tunable.BlockSize - 1) / tunable.BlockSize * tunable.BlockSize, 1, 1};

    const size_t realGridSize =
        InvariantDimVectorLoad ? GridSize / InvariantDimVectorSize : GridSize;
    const std::vector<size_t> vgd2 = {realGridSize * tunable.BlockSize, 1, 1};

    std::string algo_name = "dynamic_generic_reduction";

    std::string param = " -std=c++17 ";
    std::string network_config;

    param += get_arch_specific_compiler_flag(handle);

    param += get_definition_string_from_types<TSrc, TComp, TDst>();

    if(!reduceAllDims)
        param += " -DCK_PARAM_NUM_TOREDUCE_DIMS=" + std::to_string(mergedToReduceDims);

    param += " -DCK_PARAM_REDUCE_OP=" + std::to_string(static_cast<int>(mapReduceOpId(reduceOp)));

    param += get_definition_string_from_options(nanPropaOpt, reduceIndicesOpt);

    param += " -DCK_PARAM_IN_DIMS=" + std::to_string(mergedInvariantDims + mergedToReduceDims);
    param += " -DCK_PARAM_OUT_DIMS=";
    param += reduceAllDims ? "1" : std::to_string(mergedOutDims);

    network_config =
        get_basic_network_config_string(reduceImpl, useGlobalAtomicAdd) +
        get_network_config_string_from_types<TSrc, TComp, TDst>() + "_" +
        get_network_config_string_from_tunable(reduceImpl, &tunable, useGlobalAtomicAdd) + "_";

    network_config += std::to_string(static_cast<int>(mapReduceOpId(reduceOp))) + "_";
    network_config += get_network_config_string_from_options(nanPropaOpt, reduceIndicesOpt);

    network_config += "I" + std::to_string(mergedInvariantDims + mergedToReduceDims) + "_";

    network_config += "RED";
    network_config += std::to_string(mergedToReduceDims) + "_";

    std::vector<float> kernel1_times;
    std::vector<float> kernel2_times;
    std::vector<float> kernel3_times;
    std::vector<float> kernel4_times;

    size_t total_transfer_bytes = 0;
    float total_transfer_time   = 0;

    for(int i = 0; i < nrepeat; ++i)
    {
        KernelTimer timer1, timer2;
        auto use_padding = get_padding_need(reduceImpl,
                                            invariantLength,
                                            toReduceLength,
                                            GridSize,
                                            tunable.BlockSize,
                                            handle->GetWavefrontWidth(),
                                            BlkGroupSize,
                                            &tunable);

        std::string param1 = param +
                             " -DCK_PARAM_SRC2D_PADDING=" + std::to_string(use_padding.first) +
                             " -DCK_PARAM_DST1D_PADDING=" + std::to_string(use_padding.second);

        param1 += get_definition_string_from_tunable(reduceImpl, &tunable, useGlobalAtomicAdd);

        param1 += " -DCK_PARAM_REDUCE_DIM_VECTOR_SIZE=" + std::to_string(ReduceDimVectorSize);
        param1 += " -DCK_PARAM_INVARIANT_DIM_VECTOR_SIZE=" +
                  std::to_string(InvariantDimVectorLoad ? InvariantDimVectorSize : 1);

        std::string program_name1 =
            get_kernel_file_name(false, reduceImpl, reduceAllDims, useGlobalAtomicAdd);
        std::string kernel_name1     = "gridwise_generic_reduce_1_prepare";
        std::string network_config_1 = network_config + "_1_P" + std::to_string(reduceImpl) +
                                       std::to_string(use_padding.first) +
                                       std::to_string(use_padding.second);

        timer1.Start();

        if(!reduceAllDims)
            handle->AddKernel(
                algo_name, network_config_1, program_name1, kernel_name1, vld, vgd1, param1)(
                GridSize,
                BlkGroupSize,
                p_inLengths[0],
                p_inLengths[1],
                p_inLengths[2],
                p_inLengths[3],
                p_inLengths[4],
                p_inLengths[5],
                p_inStrides[0],
                p_inStrides[1],
                p_inStrides[2],
                p_inStrides[3],
                p_inStrides[4],
                p_inStrides[5],
                p_outLengths[0],
                p_outLengths[1],
                p_outLengths[2],
                p_outLengths[3],
                p_outLengths[4],
                p_outLengths[5],
                p_outStrides[0],
                p_outStrides[1],
                p_outStrides[2],
                p_outStrides[3],
                p_outStrides[4],
                p_outStrides[5],
                workspace.GetDeviceBuffer());
        else
            handle->AddKernel(
                algo_name, network_config_1, program_name1, kernel_name1, vld, vgd1, param1)(
                GridSize,
                BlkGroupSize,
                p_inLengths[0],
                p_inLengths[1],
                p_inLengths[2],
                p_inLengths[3],
                p_inLengths[4],
                p_inLengths[5],
                p_inStrides[0],
                p_inStrides[1],
                p_inStrides[2],
                p_inStrides[3],
                p_inStrides[4],
                p_inStrides[5],
                workspace.GetDeviceBuffer());

        timer1.End();

        if(reduceImpl == ReductionMethod_t::MultiBlock && useGlobalAtomicAdd)
        {
            auto tmpTimeVal = timer1.GetElapsedTime();

            kernel_name1     = "gridwise_generic_set_out_buffer";
            network_config_1 = "set_out_buffer";

            float zeroVal = 0.0f;

            timer1.Start();
            handle->AddKernel(
                algo_name, network_config_1, program_name1, kernel_name1, vld, vgd1_s, param1)(
                zeroVal, out_dev_buf.GetDeviceBuffer(), workspace.GetDeviceBuffer());
            timer1.End();

            tmpTimeVal += timer1.GetElapsedTime();

            kernel1_times.push_back(tmpTimeVal);
        }
        else
            kernel1_times.push_back(timer1.GetElapsedTime());

        kernel_name1     = "gridwise_generic_reduce_1";
        network_config_1 = network_config + "_1" + std::to_string(reduceImpl) +
                           std::to_string(use_padding.first) + std::to_string(use_padding.second);

        timer2.Start();
        handle->AddKernel(
            algo_name, network_config_1, program_name1, kernel_name1, vld, vgd2, param1)(
            origReduceLen,
            BlkGroupSize,
            alpha,
            in_dev_buf.GetDeviceBuffer(),
            beta,
            out_dev_buf.GetDeviceBuffer(),
            workspace.GetDeviceBuffer(),
            ws_buf2_bytes_offset,
            indices_dev_buf.GetDeviceBuffer());
        timer2.End();

        kernel2_times.push_back(timer2.GetElapsedTime());

        total_transfer_bytes = (size_t)invariantLength * toReduceLength * sizeof(TSrc);

        // Need secondary reduction only when AtomicAdd was not used by the first-time reduction
        if(reduceImpl == ReductionMethod_t::MultiBlock && !useGlobalAtomicAdd)
        {
            auto toReduceLength_2 = BlkGroupSize;
            int GridSize_2 =
                static_cast<int>(configurator.getGridSize_2(invariantLength, toReduceLength_2));
            const std::vector<size_t> vgd2_2 = {
                static_cast<size_t>(GridSize_2) * tunable.BlockSize, size_t{1}, size_t{1}};
            auto reduceImpl2 = configurator.GetReductionMethod_2(invariantLength, toReduceLength_2);
            auto tunable2    = getDefaultTunable(reduceImpl2);
            auto use_padding2 = get_padding_need(reduceImpl2,
                                                 invariantLength,
                                                 toReduceLength_2,
                                                 GridSize_2,
                                                 tunable.BlockSize,
                                                 handle->GetWavefrontWidth(),
                                                 BlkGroupSize,
                                                 &tunable2);

            std::string param2 = param +
                                 " -DCK_PARAM_SRC2D_PADDING=" + std::to_string(use_padding2.first) +
                                 " -DCK_PARAM_DST1D_PADDING=" + std::to_string(use_padding2.second);

            param2 += get_definition_string_from_tunable(reduceImpl2, &tunable2);

            param2 += " -DCK_PARAM_REDUCE_DIM_VECTOR_SIZE=" +
                      std::to_string(get_dim_vector_size<TSrc>(toReduceLength_2, 1));

            std::string program_name2    = get_kernel_file_name(true, reduceImpl2, reduceAllDims);
            std::string kernel_name2     = "gridwise_generic_reduce_2_prepare";
            std::string network_config_2 = network_config + "_2_P" + std::to_string(reduceImpl2) +
                                           std::to_string(use_padding2.first) +
                                           std::to_string(use_padding2.second);

            timer1.Start();

            if(!reduceAllDims)
                handle->AddKernel(
                    algo_name, network_config_2, program_name2, kernel_name2, vld, vgd1, param2)(
                    GridSize_2,
                    BlkGroupSize,
                    p_outLengths[0],
                    p_outLengths[1],
                    p_outLengths[2],
                    p_outLengths[3],
                    p_outLengths[4],
                    p_outLengths[5],
                    p_outStrides[0],
                    p_outStrides[1],
                    p_outStrides[2],
                    p_outStrides[3],
                    p_outStrides[4],
                    p_outStrides[5],
                    workspace.GetDeviceBuffer());
            else
                handle->AddKernel(
                    algo_name, network_config_2, program_name2, kernel_name2, vld, vgd1, param2)(
                    GridSize_2, BlkGroupSize, workspace.GetDeviceBuffer());

            timer1.End();

            kernel3_times.push_back(timer1.GetElapsedTime());

            kernel_name2     = "gridwise_generic_reduce_2";
            network_config_2 = network_config + "_2" + std::to_string(reduceImpl2) +
                               std::to_string(use_padding2.first) +
                               std::to_string(use_padding2.second);

            timer2.Start();
            handle->AddKernel(
                algo_name, network_config_2, program_name2, kernel_name2, vld, vgd2_2, param2)(
                origReduceLen,
                alpha,
                in_dev_buf.GetDeviceBuffer(),
                beta,
                out_dev_buf.GetDeviceBuffer(),
                workspace.GetDeviceBuffer(),
                ws_buf2_bytes_offset,
                indices_dev_buf.GetDeviceBuffer());
            timer2.End();

            kernel4_times.push_back(timer2.GetElapsedTime());

            total_transfer_bytes += (size_t)invariantLength * toReduceLength_2 *
                                    (need_indices ? (sizeof(TSrc) + sizeof(int)) : sizeof(TSrc)) *
                                    2;
        };

        total_transfer_bytes +=
            (size_t)invariantLength * (need_indices ? (sizeof(TDst) + sizeof(int)) : sizeof(TDst));
    }

    {
        auto ave_time1 = Driver::get_effective_average(kernel1_times);
        auto ave_time2 = Driver::get_effective_average(kernel2_times);

        total_transfer_time += ave_time2;

        if(reduceImpl == ReductionMethod_t::MultiBlock && !useGlobalAtomicAdd)
        {
            auto ave_time3 = Driver::get_effective_average(kernel3_times);
            auto ave_time4 = Driver::get_effective_average(kernel4_times);

            total_transfer_time += ave_time4;

            std::cout << "Average time : " << ave_time1 + ave_time2 + ave_time3 + ave_time3
                      << " ms(" << ave_time1 + ave_time3 << ", " << ave_time2 + ave_time4 << ")"
                      << std::endl;
            std::cout << "Average transfer rate : "
                      << total_transfer_bytes * 0.000001f / total_transfer_time << " GBytes/second"
                      << std::endl;
        }
        else
        {
            std::cout << "Average time : " << ave_time1 + ave_time2 << " ms(" << ave_time1 << ", "
                      << ave_time2 << ")" << std::endl;
            std::cout << "Average transfer rate : "
                      << total_transfer_bytes * 0.000001f / total_transfer_time << " GBytes/second"
                      << std::endl;
        };
    };

    // copy result back to host
    out_dev_buf.FromDevice(out.mData.data());

    if(need_indices)
        indices_dev_buf.FromDevice(out_indices.mData.data());
}

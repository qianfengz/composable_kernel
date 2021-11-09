#include "device.hpp"
#include "host_tensor.hpp"

#include "online_reduce_common.hpp"
#include "reduce_tunables.hpp"

#include "handle.hpp"

#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <array>

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

static int maxVectorSizeForType(appDataType_t t)
{
    using ck::DataTypeEnum_t;

    switch(t)
    {
    case appFloat: return 4;
    case appDouble: return 2;
    case appHalf: return 8;
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

    if(reduceImpl == ReductionMethod_t::MultiBlock)
    {
        outs << pt->dim0_thread_cluster_length << "_" << pt->dim0_thread_slice_length << "_";
        outs << pt->dim1_thread_cluster_length << "_" << pt->dim1_thread_slice_length << "_";
        outs << pt->reordered_thread_clusters;
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

    outs << " -DCK_PARAM_DIM0_THREAD_CLUSTER_LENGTH=" << pt->dim0_thread_cluster_length;
    outs << " -DCK_PARAM_DIM0_THREAD_SLICE_LENGTH=" << pt->dim0_thread_slice_length;
    outs << " -DCK_PARAM_DIM1_THREAD_CLUSTER_LENGTH=" << pt->dim1_thread_cluster_length;
    outs << " -DCK_PARAM_DIM1_THREAD_SLICE_LENGTH=" << pt->dim1_thread_slice_length;
    outs << " -DCK_PARAM_REORDER_THREAD_CLUSTERS=" << pt->reordered_thread_clusters;

    return (outs.str());
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
        GredUpperNumBlocksPerReduction        = 2; // used by indiced reduction

        numWarpsPerBlock = blockSize / warpSize;

        // assume 4 SIMDS per Compute Unit
        leastBlocksForOccupancy = numMaxCUs * 4 / numWarpsPerBlock;

        occupancyFactor    = 2;
        maxThreadSliceSize = 64;
    };

    int blockSize_;
    int warpSize_;
    int numWarpsPerBlock;
    int numMaxCUs_;
    // The number of blocks needed to let each SIMD has at least one warp
    int leastBlocksForOccupancy;
    // Number of active warps assigned to each SIMD
    int occupancyFactor;
    int maxThreadSliceSize;

    std::size_t GredDirectThreadWiseUpperReductionLen;
    std::size_t GredDirectWarpWiseUpperReductionLen;
    std::size_t GredBlockWiseUpperReductionLen;
    std::size_t GredUpperNumBlocksPerReduction;

    template <typename TSrc>
    std::tuple<tunable_generic_2d_reduction, int> getDefaultConfig(ReductionMethod_t reduceImpl,
                                                                   std::size_t invariantLength,
                                                                   std::size_t toReduceLength)
    {
        appDataType_t TSrcId = Driver::get_type_enum_from_type<TSrc>();

        if(reduceImpl == ReductionMethod_t::MultiBlock)
            throw std::runtime_error("This is interface should only be used with ThreadWise, "
                                     "WarpWise and MultiBlock reduction method!");

        tunable_generic_2d_reduction tunable;

        tunable.BlockSize                  = this->blockSize_;
        tunable.dim0_thread_cluster_length = 1;
        tunable.dim0_thread_slice_length   = 1;
        tunable.dim1_thread_cluster_length = this->blockSize_;

        tunable.reordered_thread_clusters = false;

        int dim1_slice_len = 1;

        if(reduceImpl == ReductionMethod_t::DirectThreadWise)
        {
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;

                if(test_slice_len <= toReduceLength &&
                   test_slice_len <= maxVectorSizeForType(TSrcId))
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };
        }
        else if(reduceImpl == ReductionMethod_t::DirectWarpWise)
        {
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;
                int warp_tile_len  = warpSize_ * test_slice_len;

                if(warp_tile_len <= toReduceLength &&
                   test_slice_len <= maxVectorSizeForType(TSrcId))
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };
        }
        else
        { //  reduceImpl == ReductionMethod_t::BlockWise
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;
                int block_tile_len = blockSize_ * test_slice_len;

                if(block_tile_len <= toReduceLength &&
                   test_slice_len <= maxVectorSizeForType(TSrcId))
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };
        }

        tunable.dim1_thread_slice_length = dim1_slice_len;

        int gridSize;

        if(reduceImpl == ReductionMethod_t::DirectThreadWise)
        {
            gridSize = (invariantLength + this->blockSize_ - 1) / this->blockSize_;
        }
        else if(reduceImpl == ReductionMethod_t::DirectWarpWise)
        {
            gridSize = (invariantLength + this->numWarpsPerBlock - 1) / this->numWarpsPerBlock;
        }
        else //  reduceImpl == ReductionMethod_t::BlockWise
            gridSize = invariantLength;

        return std::make_tuple(tunable, gridSize);
    };

    template <typename TSrc>
    std::tuple<tunable_generic_2d_reduction, int, int>
    getConfigForMultiBlock(bool need_indices,
                           std::size_t invariantLength,
                           std::size_t toReduceLength,
                           int invariant_dim_len,
                           bool invariantDimIsFastest)
    {
        appDataType_t TSrcId = Driver::get_type_enum_from_type<TSrc>();

        tunable_generic_2d_reduction tunable;

        tunable.BlockSize = this->blockSize_;

        if(invariantDimIsFastest)
        { // invariant dimension is the fastest
            tunable.reordered_thread_clusters = true;

            int dim0_cluster_len = 64;
            // get max cluster_length that can divide invariant_dim_length completely
            while(invariant_dim_len % dim0_cluster_len != 0)
                dim0_cluster_len /= 2;

            int dim0_slice_len     = 1;
            int leastDim1BlockTile = this->blockSize_ / dim0_cluster_len * 1;
            int maxBlkGroupSize    = (toReduceLength + leastDim1BlockTile - 1) / leastDim1BlockTile;

            // attempt bigger lengths for dim0_slice_len
            while(true)
            {
                int test_slice_len = dim0_slice_len * 2;
                int test_tile_size = dim0_cluster_len * test_slice_len;

                if((test_slice_len <= maxVectorSizeForType(TSrcId)) &&
                   invariant_dim_len % test_tile_size == 0 &&
                   (invariantLength / test_tile_size * maxBlkGroupSize) >=
                       this->leastBlocksForOccupancy * this->occupancyFactor)
                    dim0_slice_len = test_slice_len;
                else
                    break;
            }

            // check slice_length of 11, 7, 5, 3
            if(dim0_slice_len == 1)
            {
                static const std::array<int, 4> test_slice_lengths = {11, 7, 5, 3};
                for(const auto test_slice_len : test_slice_lengths)
                {
                    int test_tile_size = dim0_cluster_len * test_slice_len;

                    if(invariant_dim_len % test_tile_size == 0 &&
                       (invariantLength / test_tile_size * maxBlkGroupSize) >=
                           this->leastBlocksForOccupancy * this->occupancyFactor)
                    {
                        dim0_slice_len = test_slice_len;
                        break;
                    }
                }
            }

            // tune the tile assignment between thread cluster and thread slice
            while(dim0_cluster_len > 8 && dim0_slice_len < maxVectorSizeForType(TSrcId))
            {
                dim0_cluster_len /= 2;
                dim0_slice_len *= 2;
            }

            int dim0_tile_size = dim0_cluster_len * dim0_slice_len;
            int dim1_slice_len = 1;
            std::size_t gridSize;
            int blkGroupSize;
            int iterations = 1;

            // try to get big dim1_slice_len as much as possible
            if(need_indices)
            {
                // For indiced reduction, we try to use bigger dim1_slice_length to reduce the
                // number of slice access iterations since each iteration requires a reduction to
                // keep the indices consistent
                while(true)
                {
                    int test_slice_len = dim1_slice_len * 2;

                    int test_blkGroupSize =
                        (toReduceLength + (this->blockSize_ / dim0_cluster_len * test_slice_len) -
                         1) /
                        (this->blockSize_ / dim0_cluster_len * test_slice_len);

                    if(dim0_slice_len * test_slice_len <= this->maxThreadSliceSize &&
                       test_blkGroupSize > 1 &&
                       (invariantLength / dim0_tile_size * test_blkGroupSize) >=
                           this->leastBlocksForOccupancy * this->occupancyFactor)
                        dim1_slice_len = test_slice_len;
                    else
                        break;
                };

                while(true)
                {
                    int test_iterations = iterations + 1;

                    int test_blkGroupSize =
                        (toReduceLength +
                         (this->blockSize_ / dim0_cluster_len *
                          (dim1_slice_len * test_iterations)) -
                         1) /
                        (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * test_iterations));

                    if((invariantLength / dim0_tile_size * test_blkGroupSize) <
                       this->leastBlocksForOccupancy * this->occupancyFactor)
                        break;

                    if(test_blkGroupSize >= this->GredUpperNumBlocksPerReduction)
                        iterations = test_iterations;
                    else
                        break;
                };

                blkGroupSize =
                    (toReduceLength +
                     (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * iterations)) - 1) /
                    (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * iterations));
            }
            else
            {
                // For non-indiced reduction, we try to reduce the size of BlockGroup used by single
                // reduction
                while(true)
                {
                    int test_iterations = iterations + 1;
                    int test_blkGroupSize =
                        (toReduceLength +
                         (this->blockSize_ / dim0_cluster_len *
                          (dim1_slice_len * test_iterations)) -
                         1) /
                        (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * test_iterations));

                    if(test_blkGroupSize > 1 &&
                       (invariantLength / dim0_tile_size * test_blkGroupSize) >=
                           this->leastBlocksForOccupancy * this->occupancyFactor)
                        iterations = test_iterations;
                    else
                        break;
                };

                blkGroupSize =
                    (toReduceLength +
                     (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * iterations)) - 1) /
                    (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * iterations));
            }

            gridSize = invariantLength / dim0_tile_size * blkGroupSize;

            tunable.dim0_thread_cluster_length = dim0_cluster_len;
            tunable.dim0_thread_slice_length   = dim0_slice_len;
            tunable.dim1_thread_cluster_length = this->blockSize_ / dim0_cluster_len;
            tunable.dim1_thread_slice_length   = dim1_slice_len;

            return (std::make_tuple(tunable, gridSize, blkGroupSize));
        }
        else
        { // reduced dimension is the fastest
            tunable.reordered_thread_clusters  = false;
            tunable.dim0_thread_cluster_length = 1;
            tunable.dim0_thread_slice_length   = 1;

            tunable.dim1_thread_cluster_length = this->blockSize_;

            int dim1_slice_len = 1;

            // try to get big dim1_slice_len as much as possible
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;

                int test_blkGroupSize = (toReduceLength + (this->blockSize_ * test_slice_len) - 1) /
                                        (this->blockSize_ * test_slice_len);

                if(test_slice_len <= 64 && test_blkGroupSize > 1 &&
                   (invariantLength * test_blkGroupSize) >=
                       this->leastBlocksForOccupancy * this->occupancyFactor)
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };

            int gridSize;
            int blkGroupSize;
            int iterations = 1;

            // try to get big dim1_slice_len as much as possible
            if(need_indices)
            {
                // For indiced reduction, we try to use bigger dim1_slice_length to reduce the
                // number of slice access iterations since each iteration requires a reduction to
                // keep the indices consistent
                while(true)
                {
                    int test_slice_len = dim1_slice_len * 2;

                    int test_blkGroupSize =
                        (toReduceLength + (this->blockSize_ * test_slice_len) - 1) /
                        (this->blockSize_ * test_slice_len);

                    if(test_slice_len <= this->maxThreadSliceSize && test_blkGroupSize > 1 &&
                       (invariantLength * test_blkGroupSize) >=
                           this->leastBlocksForOccupancy * this->occupancyFactor)
                        dim1_slice_len = test_slice_len;
                    else
                        break;
                };

                while(true)
                {
                    int test_iterations = iterations + 1;

                    int test_blkGroupSize =
                        (toReduceLength + (this->blockSize_ * (dim1_slice_len * test_iterations)) -
                         1) /
                        (this->blockSize_ * (dim1_slice_len * test_iterations));

                    if((invariantLength * test_blkGroupSize) <
                       this->leastBlocksForOccupancy * this->occupancyFactor)
                        break;

                    if(test_blkGroupSize >= this->GredUpperNumBlocksPerReduction)
                        iterations = test_iterations;
                    else
                        break;
                };

                blkGroupSize =
                    (toReduceLength + (this->blockSize_ * (dim1_slice_len * iterations)) - 1) /
                    (this->blockSize_ * (dim1_slice_len * iterations));
            }
            else
            {
                // For non-indiced reduction, we try to reduce the size of BlockGroup used by single
                // reduction
                while(true)
                {
                    int test_iterations = iterations + 1;
                    int test_blkGroupSize =
                        (toReduceLength + (this->blockSize_ * (dim1_slice_len * test_iterations)) -
                         1) /
                        (this->blockSize_ * (dim1_slice_len * test_iterations));

                    if(test_blkGroupSize > 1 &&
                       (invariantLength * test_blkGroupSize) >=
                           this->leastBlocksForOccupancy * this->occupancyFactor)
                        iterations = test_iterations;
                    else
                        break;
                };

                blkGroupSize =
                    (toReduceLength + (this->blockSize_ * (dim1_slice_len * iterations)) - 1) /
                    (this->blockSize_ * (dim1_slice_len * iterations));
            }

            gridSize = invariantLength * blkGroupSize;

            tunable.dim1_thread_slice_length = dim1_slice_len;

            return (std::make_tuple(tunable, gridSize, blkGroupSize));
        }
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

    std::size_t getWorkspaceSize(bool useGlobalAtomicAdd,
                                 ReductionMethod_t reduceImpl,
                                 size_t invariantLength,
                                 int blkGroupSize) const
    {
        if(reduceImpl == ReductionMethod_t::MultiBlock && !useGlobalAtomicAdd)
            return (invariantLength * blkGroupSize);
        return (0);
    };

    std::size_t getGridSize_2(ReductionMethod_t reduceImpl,
                              std::size_t invariantLength,
                              std::size_t toReduceLength) const
    {
        (void)toReduceLength;

        if(reduceImpl == ReductionMethod_t::DirectThreadWise)
        {
            return ((invariantLength + blockSize_ - 1) / blockSize_);
        }
        else if(reduceImpl == ReductionMethod_t::DirectWarpWise)
        {
            return ((invariantLength + numWarpsPerBlock - 1) / numWarpsPerBlock);
        }
        else if(reduceImpl == ReductionMethod_t::BlockWise)
        {
            return (invariantLength);
        }
        else
            throw std::runtime_error("Invalid reduction method used!");
    };

    ReductionMethod_t getReductionMethod_2(std::size_t invariantLength,
                                           std::size_t toReduceLength) const
    {
        (void)invariantLength;

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
        copySliceLen = tunable->dim1_thread_cluster_length * tunable->dim1_thread_slice_length;
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

static std::string get_kernel_file_name(const bool isSecondCall,
                                        const ReductionMethod_t reduceImpl,
                                        const bool allDimsReduced,
                                        const bool useGlobalAtomicAdd = false)
{
    std::ostringstream outs;

    if(reduceImpl == ReductionMethod_t::MultiBlock)
    {
        outs << "gridwise_generic_reduction_" << getReductionMethodStr(reduceImpl);

        if(allDimsReduced)
            outs << "_reduce_all_dims";
        else
            outs << "_reduce_partial_dims";

        if(useGlobalAtomicAdd)
            outs << "_atomic_add";

        outs << "_gc.cpp";
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
static int get_dim_max_vector_size(int dimLength, int dimStride)
{
    appDataType_t dataTypeId = Driver::get_type_enum_from_type<dataType>();
    int len                  = maxVectorSizeForType(dataTypeId);

    // not fastest dim
    if(dimStride != 1)
        return (1);

    while(len != 1)
    {
        if(dimLength % len == 0)
            break;

        len /= 2;
    }

    return (len);
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
        get_dim_max_vector_size<TSrc>(p_inLengths[mergedInvariantDims + mergedToReduceDims - 1],
                                      p_inStrides[mergedInvariantDims + mergedToReduceDims - 1]);
    int InvariantDimVectorSize = get_dim_max_vector_size<TSrc>(
        p_inLengths[mergedInvariantDims - 1], p_inStrides[mergedInvariantDims - 1]);

    // either the invariant lowest dimension or the toReduce lowest dimension is the fastest one
    // among all dimensions
    assert(p_inStrides[mergedInvariantDims + mergedToReduceDims - 1] == 1 ||
           p_inStrides[mergedInvariantDims - 1] == 1);

    bool InvariantDimIsFastest = p_inStrides[mergedInvariantDims - 1] == 1 ? true : false;

    bool need_indices = (reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES) &&
                        (reduceOp == REDUCE_TENSOR_MIN || reduceOp == REDUCE_TENSOR_MAX ||
                         reduceOp == REDUCE_TENSOR_AMAX);

    ReductionMethod_t reduceImpl = configurator.getReductionMethod(invariantLength, toReduceLength);
    int GridSize;
    int BlkGroupSize = 0;
    tunable_generic_2d_reduction tunable;

    if(reduceImpl == ReductionMethod_t::MultiBlock)
    {
        std::tie(tunable, GridSize, BlkGroupSize) =
            configurator.getConfigForMultiBlock<TSrc>(need_indices,
                                                      invariantLength,
                                                      toReduceLength,
                                                      p_inLengths[mergedInvariantDims - 1],
                                                      InvariantDimIsFastest);
    }
    else
    {
        std::tie(tunable, GridSize) =
            configurator.getDefaultConfig<TSrc>(reduceImpl, invariantLength, toReduceLength);
    }

    const bool useGlobalAtomicAdd =
        useGlobalAtomicAddReduce(reduceOp, Driver::get_type_enum_from_type<TDst>(), beta);

    auto workspace_size = configurator.getWorkspaceSize(
        useGlobalAtomicAdd, reduceImpl, invariantLength, BlkGroupSize);

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

    const std::vector<size_t> vld    = {static_cast<size_t>(tunable.BlockSize), 1, 1};
    const std::vector<size_t> vgd1   = {static_cast<size_t>(tunable.BlockSize), 1, 1};
    const std::vector<size_t> vgd1_s = {
        (invariantLength + tunable.BlockSize - 1) / tunable.BlockSize * tunable.BlockSize, 1, 1};

    const std::vector<size_t> vgd2 = {(size_t)GridSize * tunable.BlockSize, 1, 1};

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
        param1 += " -DCK_PARAM_INVARIANT_DIM_VECTOR_SIZE=" + std::to_string(InvariantDimVectorSize);

        if(InvariantDimIsFastest)
            param1 += " -DCK_PARAM_INVARIANT_DIM_IS_FASTEST=1";
        else
            param1 += " -DCK_PARAM_INVARIANT_DIM_IS_FASTEST=0";

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
            auto reduceImpl2 = configurator.getReductionMethod_2(invariantLength, toReduceLength_2);
            tunable_generic_2d_reduction tunable2;
            int GridSize_2;

            std::tie(tunable2, GridSize_2) =
                configurator.getDefaultConfig<TSrc>(reduceImpl2, invariantLength, toReduceLength_2);

            const std::vector<size_t> vgd2_2 = {
                (size_t)GridSize_2 * tunable.BlockSize, size_t{1}, size_t{1}};

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
                      std::to_string(get_dim_max_vector_size<TSrc>(toReduceLength_2, 1));

            param2 += " -DCK_PARAM_INVARIANT_DIM_IS_FASTEST=0";

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

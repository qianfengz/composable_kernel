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

static int gcd(int x, int y)
{
    if(x < 0)
    {
        return gcd(-x, y);
    }
    else if(y < 0)
    {
        return gcd(x, -y);
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y);
    }
    else
    {
        return gcd(x, y % x);
    }
};

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

    outs << pt->dim0_thread_cluster_size << "_" << pt->dim0_thread_slice_size << "_";
    outs << pt->dim1_thread_cluster_size << "_" << pt->dim1_thread_slice_size << "_";

    return (outs.str());
};

static std::string get_definition_string_from_tunable(const tunable_generic_2d_reduction* pt)
{
    std::ostringstream outs;

    outs << " -DCK_PARAM_BLOCKSIZE=" << pt->BlockSize;

    outs << " -DCK_PARAM_DIM0_THREAD_CLUSTER_SIZE=" << pt->dim0_thread_cluster_size;
    outs << " -DCK_PARAM_DIM0_THREAD_SLICE_SIZE=" << pt->dim0_thread_slice_size;
    outs << " -DCK_PARAM_DIM1_THREAD_CLUSTER_SIZE=" << pt->dim1_thread_cluster_size;
    outs << " -DCK_PARAM_DIM1_THREAD_SLICE_SIZE=" << pt->dim1_thread_slice_size;

    return (outs.str());
};

struct ReductionKernelConfigurator
{
    ReductionKernelConfigurator() = default;

    ReductionKernelConfigurator(int blockSize, int warpSize, int numMaxCUs)
        : blockSize_(blockSize), warpSize_(warpSize), numMaxCUs_(numMaxCUs)
    {
        GredDirectThreadWiseUpperReductionLen = warpSize;
        GredBlockWiseUpperReductionLen        = blockSize * 4;
        GredLeastNumBlocksPerReduction        = 2; // used by indiced reduction

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
    std::size_t GredBlockWiseUpperReductionLen;
    std::size_t GredLeastNumBlocksPerReduction;

    template <appDataType_t TSrcId>
    std::tuple<tunable_generic_2d_reduction, int>
    getConfigForThreadWise(bool need_indices, std::size_t dim0_total_size, std::size_t dim1_total_size, int dim0_lowest_size, int dim1_lowest_size, int vectorDim)
    {
        tunable_generic_2d_reduction tunable;

        tunable.BlockSize = this->blockSize_;
	tunable.dim0_thread_cluster_size = this->blockSize_; 
	tunable.dim1_thread_cluster_size = 1; 

        if(vectorDim==0)  // dim0 is the vector dim
	{
            int dim0_slice_len = 1; 

            // Try to let dim0_slice_len as big as possible 
            while(true)
            {
                int test_slice_len = dim0_slice_len * 2;
                int test_tile_size = this->blockSize_ * test_slice_len;
                int test_grid_size = (dim0_total_size + test_tile_size - 1) / test_tile_size;

                if((test_slice_len <= maxVectorSizeForType(TSrcId)) && dim0_lowest_size % test_slice_len == 0 &&
                    test_grid_size >= this->leastBlocksForOccupancy * this->occupancyFactor)
                    dim0_slice_len = test_slice_len;
                else
                    break;
            };

            int dim1_slice_len = 1;

            // Try to let dim1_slice_len as big as possible
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;
                int test_tile_len = test_slice_len;

                if(test_slice_len <= maxVectorSizeForType(TSrcId) && dim1_lowest_size % test_slice_len == 0 &&
                    dim0_slice_len * test_slice_len <= this->maxThreadSliceSize && test_tile_len < dim1_total_size * 2)
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };

	    tunable.dim0_thread_slice_size = dim0_slice_len; 
	    tunable.dim1_thread_slice_size = dim1_slice_len; 
	}
	else   // dim1 is the vector dim
	{
            int dim0_slice_len = 1; 
            int dim1_slice_len = 1; 

            // Try to let dim1_slice_len as big as possible
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;
                int test_tile_len = test_slice_len;

                if(test_slice_len <= maxVectorSizeForType(TSrcId) && dim1_lowest_size % test_slice_len == 0 &&
                    dim0_slice_len * test_slice_len <= this->maxThreadSliceSize && test_tile_len < dim1_total_size * 2)
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };

            tunable.dim0_thread_slice_size = dim0_slice_len;
            tunable.dim1_thread_slice_size = dim1_slice_len;
	}; 

        int dim0_tile_size = tunable.dim0_thread_cluster_size * tunable.dim0_thread_slice_size;
        int gridSize = (dim0_total_size + dim0_tile_size - 1) / dim0_tile_size;

        return(std::make_tuple(tunable, gridSize));
    };

    template <appDataType_t TSrcId>
    std::tuple<tunable_generic_2d_reduction, int>
    getConfigForBlockWise(bool need_indices, std::size_t dim0_total_size, std::size_t dim1_total_size, int dim0_lowest_size, int dim1_lowest_size, int vectorDim)
    {
        tunable_generic_2d_reduction tunable;

        tunable.BlockSize = this->blockSize_;

        if(vectorDim==0)  // dim0 is the vector dim
	{ 
            int dim0_cluster_len = this->blockSize_ / 2;
             
            // Try to let dim0_cluster_len as small as possible to get enough grid size 
            while(true)
            {
		int test_cluster_len = dim0_cluster_len / 2; 
                int test_grid_size = (dim0_total_size + test_cluster_len - 1) / test_cluster_len; 

		if (test_cluster_len < 1 || test_grid_size < this->leastBlocksForOccupancy * this->occupancyFactor) 
	            break; 

		dim0_cluster_len = test_cluster_len; 
            }; 		    

            int dim0_slice_len = 1; 

	    // Try to let dim0_slice_len as big as possible 
	    while(true)
	    {
                int test_slice_len = dim0_slice_len * 2;
                int test_tile_size = dim0_cluster_len * test_slice_len;
                int test_grid_size = (dim0_total_size + test_tile_size -1) / test_tile_size;

                if((test_slice_len <= maxVectorSizeForType(TSrcId)) && dim0_lowest_size % test_slice_len == 0 &&
                    test_grid_size >= this->leastBlocksForOccupancy * this->occupancyFactor)
                    dim0_slice_len = test_slice_len;
                else
                    break;
	    }; 

            int dim1_slice_len = 1; 

	    // Try to let dim1_slice_len as big as possible
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;
                int test_tile_len = tunable.BlockSize / dim0_cluster_len * test_slice_len;

                if(test_slice_len <= maxVectorSizeForType(TSrcId) && dim1_lowest_size % test_slice_len == 0 && 
		    dim0_slice_len * test_slice_len <= this->maxThreadSliceSize && test_tile_len < dim1_total_size * 2) 
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };

            tunable.dim0_thread_cluster_size = dim0_cluster_len; 
	    tunable.dim1_thread_cluster_size = tunable.BlockSize / dim0_cluster_len; 
	    tunable.dim0_thread_slice_size = dim0_slice_len; 
	    tunable.dim1_thread_slice_size = dim1_slice_len; 
	}
	else  // dim1 is the vector dim
	{
	    int dim1_cluster_len = tunable.BlockSize; 

	    // Try to let dim1_cluster_len as small as possible
	    while(true)
	    {
		int test_cluster_len = dim1_cluster_len / 2; 
		
		if(test_cluster_len >= dim1_total_size) 
		    dim1_cluster_len = test_cluster_len; 
		else
		    break;
	    }; 

            int dim1_slice_len = 1; 

	    // Try to let dim1_slice_len as big as possible
	    while(true)
	    {
		int test_slice_len = dim1_slice_len * 2; 
                int test_tile_len = dim1_cluster_len * test_slice_len; 

		if(test_tile_len <= dim1_total_size * 2 && dim1_lowest_size % test_slice_len == 0 && 
		    test_slice_len <= maxVectorSizeForType(TSrcId) && test_slice_len <= this->maxThreadSliceSize)
		    dim1_slice_len = test_slice_len; 
		else
		    break; 
            };

	    tunable.dim1_thread_cluster_size = dim1_cluster_len; 
	    tunable.dim1_thread_slice_size = dim1_slice_len; 
            tunable.dim0_thread_cluster_size = tunable.BlockSize / dim1_cluster_len; 
	    tunable.dim0_thread_slice_size = 1; 
	}; 

        int dim0_tile_size = tunable.dim0_thread_cluster_size * tunable.dim0_thread_slice_size; 
        int gridSize = (dim0_total_size + dim0_tile_size - 1) / dim0_tile_size; 

	return(std::make_tuple(tunable, gridSize)); 
    }; 

    template <appDataType_t TSrcId>
    std::tuple<tunable_generic_2d_reduction, int, int>
    getConfigForMultiBlock(bool need_indices, std::size_t dim0_total_size, std::size_t dim1_total_size, int dim0_lowest_size, int dim1_lowest_size, int vectorDim)
    {
        tunable_generic_2d_reduction tunable;

        tunable.BlockSize = this->blockSize_;

        if(vectorDim==0)    // dim0 is the vector dim
        { 
            int dim0_cluster_len = this->blockSize_ / 2;

            // Try to let dim0_cluster_len as small as possible
            while(dim0_lowest_size % dim0_cluster_len != 0)
                dim0_cluster_len /= 2;

            int dim0_slice_len     = 1;
            int leastDim1BlockTile = this->blockSize_ / dim0_cluster_len * 1;
            int maxBlkGroupSize    = (dim1_total_size + leastDim1BlockTile - 1) / leastDim1BlockTile;

            // Try to let dim0_slice_len as big as possible
            while(true)
            {
                int test_slice_len = dim0_slice_len * 2;
                int test_tile_size = dim0_cluster_len * test_slice_len;
                int test_grid_size = dim0_total_size / test_tile_size * maxBlkGroupSize;

                if((test_slice_len <= maxVectorSizeForType(TSrcId)) &&
                   dim0_lowest_size % test_tile_size == 0 &&
                   test_grid_size >= this->leastBlocksForOccupancy * this->occupancyFactor)
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
                    int test_grid_size = dim0_total_size / test_tile_size * maxBlkGroupSize; 

                    if(dim0_lowest_size % test_tile_size == 0 && test_grid_size >= this->leastBlocksForOccupancy * this->occupancyFactor)
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

            // Try to let dim1_slice_len as big as possible
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;
                int test_tile_len = this->blockSize_ / dim0_cluster_len * test_slice_len; 
                int test_blkGroupSize = (dim1_total_size + test_tile_len - 1) / test_tile_len;
                int test_grid_size = dim0_total_size / dim0_tile_size * test_blkGroupSize;

                if(dim0_slice_len * test_slice_len <= this->maxThreadSliceSize && dim1_lowest_size % test_slice_len == 0 &&
                    test_blkGroupSize >= this->GredLeastNumBlocksPerReduction &&
                    test_grid_size >= this->leastBlocksForOccupancy * this->occupancyFactor)
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };

            // Try to let blkGroupSize as small as possible 
            while(true)
            {
                int test_iterations = iterations + 1;
                int test_tiles_size = this->blockSize_ / dim0_cluster_len * (dim1_slice_len * test_iterations); 
                int test_blkGroupSize = (dim1_total_size + test_tiles_size -1) / test_tiles_size; 
                int test_grid_size = dim0_total_size / dim0_tile_size * test_blkGroupSize;

                if(test_blkGroupSize < this->GredLeastNumBlocksPerReduction)
		    break;

                if(test_grid_size < this->leastBlocksForOccupancy * this->occupancyFactor)
                    break;

                iterations = test_iterations;
            };

            blkGroupSize =
                    (dim1_total_size +
                     (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * iterations)) - 1) /
                    (this->blockSize_ / dim0_cluster_len * (dim1_slice_len * iterations));

            gridSize = dim0_total_size / dim0_tile_size * blkGroupSize;

            tunable.dim0_thread_cluster_size = dim0_cluster_len;
            tunable.dim0_thread_slice_size   = dim0_slice_len;
            tunable.dim1_thread_cluster_size = this->blockSize_ / dim0_cluster_len;
            tunable.dim1_thread_slice_size   = dim1_slice_len;

            return (std::make_tuple(tunable, gridSize, blkGroupSize));
        }
        else  // dim1 is the vector dim
        {   
            tunable.dim0_thread_cluster_size = 1;
            tunable.dim0_thread_slice_size   = 1;

            tunable.dim1_thread_cluster_size = this->blockSize_;

            int dim1_slice_len = 1;

            // try to let dim1_slice_len as big as possible
            while(true)
            {
                int test_slice_len = dim1_slice_len * 2;
                int test_tile_len = this->blockSize_ * test_slice_len;
                int test_blkGroupSize = (dim1_total_size + test_tile_len - 1) / test_tile_len;
                int test_grid_size = dim0_total_size * test_blkGroupSize;

                if(test_slice_len <= this->maxThreadSliceSize && dim1_lowest_size % test_slice_len == 0 && 
		   test_blkGroupSize >= this->GredLeastNumBlocksPerReduction &&
                   test_grid_size >= this->leastBlocksForOccupancy * this->occupancyFactor)
                    dim1_slice_len = test_slice_len;
                else
                    break;
            };

            int gridSize;
            int blkGroupSize;
            int iterations = 1;

            // Try to let blkGroupSize as small as possible
            while(true)
            {
                int test_iterations = iterations + 1;
                int test_tiles_size = this->blockSize_ * dim1_slice_len * test_iterations; 
                int test_blkGroupSize = (dim1_total_size + test_tiles_size - 1) / test_tiles_size;
                int test_grid_size = dim0_total_size * test_blkGroupSize;

                if(test_blkGroupSize < this->GredLeastNumBlocksPerReduction)
		    break; 

                if(test_grid_size < this->leastBlocksForOccupancy * this->occupancyFactor)
                    break;

                iterations = test_iterations;
            };

            blkGroupSize =
                    (dim1_total_size + (this->blockSize_ * (dim1_slice_len * iterations)) - 1) /
                    (this->blockSize_ * (dim1_slice_len * iterations));

            gridSize = dim0_total_size * blkGroupSize;

            tunable.dim1_thread_slice_size = dim1_slice_len;

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
                    GredBlockWiseUpperReductionLen) // let one block to do each reduction
                return (ReductionMethod_t::BlockWise);
            else
                return (ReductionMethod_t::MultiBlock); // let multiple blocks to do each reduction
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

    if(isSecondCall) {
        if(allDimsReduced) 
	   outs << "gridwise_reduction_second_call_blockwise_reduce_all_dims.cpp"; 
	else
	   outs << "gridwise_reduction_second_call_blockwise_reduce_partial_dims.cpp"; 

        return(outs.str()); 
    }; 
	
    if(reduceImpl == ReductionMethod_t::MultiBlock)
    {
        if(useGlobalAtomicAdd)
            outs << "gridwise_reduction_multiblock_atomic_add";
        else 
	    outs << "gridwise_reduction_multiblock_two_call"; 

        if(allDimsReduced)
            outs << "_reduce_all_dims.cpp";
        else
            outs << "_reduce_partial_dims.cpp";
    }
    else
    if(reduceImpl == ReductionMethod_t::DirectThreadWise)
    {
        if(allDimsReduced)
            outs << "gridwise_reduction_threadwise_reduce_all_dims.cpp";
        else
            outs << "gridwise_reduction_threadwise_reduce_partial_dims.cpp";
    }
    else 
    if(reduceImpl == ReductionMethod_t::BlockWise)
    {
        if(allDimsReduced)
            outs << "gridwise_reduction_blockwise_reduce_all_dims.cpp";
        else
            outs << "gridwise_reduction_blockwise_reduce_partial_dims.cpp";
    }
    else
	throw std::runtime_error("Invalid ReductionMethod enum value!"); 

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

    int dim0_lowest_size = (mergedInvariantDims == 0)? 1 : p_inLengths[mergedInvariantDims - 1]; 
    int dim1_lowest_size = p_inLengths[mergedInvariantDims + mergedToReduceDims - 1]; 

    // either the invariant lowest dimension or the toReduce lowest dimension is the fastest one
    // among all dimensions
    assert(p_inStrides[mergedInvariantDims + mergedToReduceDims - 1] == 1 ||
           p_inStrides[mergedInvariantDims - 1] == 1);

    int vectorDim =  (mergedInvariantDims == 0)? 1 : (p_inStrides[mergedInvariantDims - 1] == 1 ? 0 : 1);  

    bool need_indices = (reduceIndicesOpt == REDUCE_TENSOR_FLATTENED_INDICES) &&
                        (reduceOp == REDUCE_TENSOR_MIN || reduceOp == REDUCE_TENSOR_MAX ||
                         reduceOp == REDUCE_TENSOR_AMAX);

    ReductionMethod_t reduceImpl = configurator.getReductionMethod(invariantLength, toReduceLength);
    int GridSize;
    int BlkGroupSize = 0;
    tunable_generic_2d_reduction tunable;

    constexpr auto TSrcId = Driver::get_type_enum_from_type<TSrc>(); 
    constexpr auto TCompId = Driver::get_type_enum_from_type<TComp>(); 

    if(reduceImpl == ReductionMethod_t::MultiBlock)
        std::tie(tunable, GridSize, BlkGroupSize) =
            configurator.getConfigForMultiBlock<TSrcId>(need_indices, invariantLength, toReduceLength, dim0_lowest_size, dim1_lowest_size, vectorDim); 
    else
    if(reduceImpl == ReductionMethod_t::DirectThreadWise)
        std::tie(tunable, GridSize) = configurator.getConfigForThreadWise<TSrcId>(need_indices, invariantLength, toReduceLength, dim0_lowest_size, dim1_lowest_size, vectorDim); 
    else
        std::tie(tunable, GridSize) = configurator.getConfigForBlockWise<TSrcId>(need_indices, invariantLength, toReduceLength, dim0_lowest_size, dim1_lowest_size, vectorDim); 

    std::cout << "gridSize = " << GridSize <<", in_dev_buf = " << in_dev_buf.GetDeviceBuffer() << std::endl; 

    const bool useGlobalAtomicAdd =
        useGlobalAtomicAddReduce(reduceOp, Driver::get_type_enum_from_type<TDst>(), beta);

    auto workspace_size = configurator.getWorkspaceSize(
        useGlobalAtomicAdd, reduceImpl, invariantLength, BlkGroupSize);

    size_t wsSizeInBytes = !need_indices
                               ? workspace_size * sizeof(TComp)
                               : workspace_size * (sizeof(TComp) + sizeof(int)) + 64 + sizeof(int);

    DeviceMem workspace(4096 + wsSizeInBytes);

    size_t indicesSizeInBytes = need_indices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

    DeviceMem indices_dev_buf(indicesSizeInBytes);

    size_t ws_buf2_bytes_offset = 0;

    if(need_indices && workspace_size > 0)
    {
        size_t byteOffset =
            static_cast<size_t>((wsSizeInBytes / (sizeof(TComp) + sizeof(int))) * sizeof(TComp));

        ws_buf2_bytes_offset = ((byteOffset + 63) / 64) * 64;
    };

    int vectorSize = (vectorDim == 0)? gcd(tunable.dim0_thread_slice_size, maxVectorSizeForType(TSrcId)) :
              	                       gcd(tunable.dim1_thread_slice_size, maxVectorSizeForType(TSrcId)); 

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

        std::string param1 = param; 

        param1 += get_definition_string_from_tunable(&tunable);

        param1 += " -DCK_PARAM_VECTOR_DIM=" + std::to_string(vectorDim); 
	param1 += " -DCK_PARAM_VECTOR_SIZE=" + std::to_string(vectorSize); 

        std::string program_name1 =
            get_kernel_file_name(false, reduceImpl, reduceAllDims, useGlobalAtomicAdd);
        std::string kernel_name1     = "gridwise_generic_reduce_1_prepare";
        std::string network_config_1 = network_config + "_1_P" + std::to_string(reduceImpl); 

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
        network_config_1 = network_config + "_1" + std::to_string(reduceImpl);

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
            auto reduceImpl2 = ReductionMethod_t::BlockWise; 
            tunable_generic_2d_reduction tunable2;
            int GridSize_2;

            std::tie(tunable2, GridSize_2) =
                configurator.getConfigForBlockWise<TCompId>(need_indices, invariantLength, toReduceLength_2, invariantLength, toReduceLength_2, 1);

            const std::vector<size_t> vgd2_2 = {
                (size_t)GridSize_2 * tunable.BlockSize, size_t{1}, size_t{1}};

            std::string param2 = param; 

            param2 += get_definition_string_from_tunable(&tunable2);

            param2 += " -DCK_PARAM_VECTOR_DIM=1"; 

	    int vectorSize2 = gcd(tunable2.dim1_thread_slice_size, maxVectorSizeForType(TCompId)); 

            param2 += " -DCK_PARAM_VECTOR_SIZE=" + std::to_string(vectorSize2); 

            std::string program_name2    = get_kernel_file_name(true, reduceImpl2, reduceAllDims);
            std::string kernel_name2     = "gridwise_generic_reduce_2_prepare";
            std::string network_config_2 = network_config + "_2_P" + std::to_string(reduceImpl2);

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
            network_config_2 = network_config + "_2" + std::to_string(reduceImpl2); 

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

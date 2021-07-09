#ifndef CONV_TUNABLES_HPP
#define CONV_TUNABLES_HPP

#include "config.hpp"

struct tunable_dyn_conv_fwd_v4r4_nchw_kcyx_nkhw
{
    ck::index_t BlockSize; // usually not tunable

    ck::index_t MPerBlock;
    ck::index_t NPerBlock;
    ck::index_t KPerBlock;

    ck::index_t M1PerThread;
    ck::index_t N1PerThread;
    ck::index_t KPerThread;

    ck::index_t M1N1ThreadClusterM10;
    ck::index_t M1N1ThreadClusterN10;
    ck::index_t M1N1ThreadClusterM11;
    ck::index_t M1N1ThreadClusterN11;

    std::array<ck::index_t, 3> ABlockTransferThreadSliceLengths_K_M0_M1;
    std::array<ck::index_t, 3> ABlockTransferThreadClusterLengths_K_M0_M1;
    std::array<ck::index_t, 3> ABlockTransferThreadClusterArrangeOrder;
    std::array<ck::index_t, 3> ABlockTransferSrcAccessOrder;
    ck::index_t ABlockTransferSrcVectorDim;
    ck::index_t ABlockTransferSrcScalarPerVector;
    ck::index_t ABlockTransferDstScalarPerVector_M1;
    bool AThreadTransferSrcResetCoordinateAfterRun;

    std::array<ck::index_t, 3> BBlockTransferThreadSliceLengths_K_N0_N1;
    std::array<ck::index_t, 3> BBlockTransferThreadClusterLengths_K_N0_N1;
    std::array<ck::index_t, 3> BBlockTransferThreadClusterArrangeOrder;
    std::array<ck::index_t, 3> BBlockTransferSrcAccessOrder;
    ck::index_t BBlockTransferSrcVectorDim;
    ck::index_t BBlockTransferSrcScalarPerVector;
    ck::index_t BBlockTransferDstScalarPerVector_N1;
    bool BThreadTransferSrcResetCoordinateAfterRun;

    std::array<ck::index_t, 6> CThreadTransferSrcDstAccessOrder;
    ck::index_t CThreadTransferSrcDstVectorDim;
    ck::index_t CThreadTransferDstScalarPerVector;
};

static tunable_dyn_conv_fwd_v4r4_nchw_kcyx_nkhw default_tunable_dyn_conv_fwd_v4r4_nchw_kcyx_nkhw = {
    256,       128,       128, 8, 4,         4,           1,
    8,         8,         2,   2, {4, 1, 1}, {2, 1, 128}, {2, 1, 0},
    {2, 1, 0}, 0,         4,   1, false,     {4, 1, 1},   {2, 1, 128},
    {0, 1, 2}, {0, 1, 2}, 2,   1, 1,         false,       {3, 4, 5, 0, 1, 2},
    5,         1};

struct tunable_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw
{
    ck::index_t BlockSize;

    ck::index_t GM1PerBlockGM11;
    ck::index_t GN1PerBlockGN11;
    ck::index_t KPerBlock;

    ck::index_t M1PerThread;
    ck::index_t N1PerThread;
    ck::index_t KPerThread;

    ck::index_t M1N1ThreadClusterM10;
    ck::index_t M1N1ThreadClusterN10;
    ck::index_t M1N1ThreadClusterM11;
    ck::index_t M1N1ThreadClusterN11;

    std::array<ck::index_t, 4> ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11;
    std::array<ck::index_t, 4> ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11;
    std::array<ck::index_t, 4> ABlockTransferThreadClusterArrangeOrder;
    std::array<ck::index_t, 4> ABlockTransferSrcAccessOrder;
    ck::index_t ABlockTransferSrcVectorDim;
    ck::index_t ABlockTransferSrcScalarPerVector;
    ck::index_t ABlockTransferDstScalarPerVector_GM11;
    bool AThreadTransferSrcResetCoordinateAfterRun;

    std::array<ck::index_t, 4> BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11;
    std::array<ck::index_t, 4> BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11;
    std::array<ck::index_t, 4> BBlockTransferThreadClusterArrangeOrder;
    std::array<ck::index_t, 4> BBlockTransferSrcAccessOrder;
    ck::index_t BBlockTransferSrcVectorDim;
    ck::index_t BBlockTransferSrcScalarPerVector;
    ck::index_t BBlockTransferDstScalarPerVector_GN11;
    bool BThreadTransferSrcResetCoordinateAfterRun;

    std::array<ck::index_t, 6> CThreadTransferSrcDstAccessOrder;
    ck::index_t CThreadTransferSrcDstVectorDim;
    ck::index_t CThreadTransferDstScalarPerVector;
};

static tunable_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw default_tunable_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw = {
    256,
    128,
    32,
    8,
    4,
    4,
    1,
    2,
    2,
    8,
    8,
    {4, 1, 1, 1},
    {2, 1, 1, 128},
    {3, 2, 1, 0},
    {3, 2, 1, 0},
    0,
    4,
    1,
    false,
    {1, 4, 1, 1},
    {8, 1, 1, 32},
    {0, 3, 2, 1},
    {0, 3, 2, 1},
    3,
    1,
    1,
    false,
    {3, 4, 5, 0, 1, 2},
    5,
    1};

static inline int
conv_hw_out_size(int hw_in_size, int leftPad, int rightPad, int dilation, int yx_size, int stride)
{
    return (hw_in_size + leftPad + rightPad - dilation * (yx_size - 1) - 1) / stride + 1;
}

<<<<<<< HEAD
struct tunable_dyn_generic_reduction
{
    ck::index_t BlockSize; 
    ck::index_t GredThreadBufferLength;
    ck::index_t GredAccessesPerThreadInBlock;
    ck::index_t GredAccessesPerThreadInWarp;
}; 

static struct tunable_dyn_generic_reduction default_tunable_dyn_generic_reduction = {256, 8, 2, 2}; 

=======
>>>>>>> master
#endif

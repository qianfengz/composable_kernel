#ifndef GENERIC_REDUCTION_TUNABLES_HPP
#define GENERIC_REDUCTION_TUNABLES_HPP

struct tunable_generic_2d_reduction
{
    ck::index_t BlockSize;

    // dim0 is invariant dimension
    ck::index_t dim0_thread_slice_length;
    ck::index_t dim0_thread_cluster_length;

    // dim 1 is to-reduce dimension
    ck::index_t dim1_thread_slice_length;
    ck::index_t dim1_thread_cluster_length;

    // true -- indicates lower Thread Id bits are assigned to dim0, upper Thread Id bits are
    // assigned to dim1, false -- indicates upper Thread Ids bits are assigned to dim0, lower Thread
    // Id bits are assigned to dim1
    bool reordered_thread_clusters;
};

#endif

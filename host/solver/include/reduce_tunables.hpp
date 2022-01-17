#ifndef GENERIC_REDUCTION_TUNABLES_HPP
#define GENERIC_REDUCTION_TUNABLES_HPP

struct tunable_generic_2d_reduction
{
    ck::index_t BlockSize;

    // dim0 is invariant dimension
    ck::index_t dim0_thread_slice_size;
    ck::index_t dim0_thread_cluster_size;

    // dim 1 is to-reduce dimension
    ck::index_t dim1_thread_slice_size;
    ck::index_t dim1_thread_cluster_size;
};

#endif

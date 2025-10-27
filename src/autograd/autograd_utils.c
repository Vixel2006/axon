#include "autograd/autograd_utils.h"
#include "logger.h"
#include "utils.h"

int get_reduced_dim(int* in_shape, int* out_shape, int in_ndim, int out_ndim)
{
    LOG_INFO("get_reduced_dim: Entering function");
    LOG_DEBUG("get_reduced_dim: in_ndim=%d, out_ndim=%d", in_ndim, out_ndim);

    int reduced_dim = -1;
    for (int i = 0; i < in_ndim; ++i)
    {
        if (out_ndim <= i || in_shape[i] != out_shape[i])
        {
            reduced_dim = i;
            break;
        }
    }

    LOG_DEBUG("get_reduced_dim: Reduced dimension: %d", reduced_dim);
    LOG_INFO("get_reduced_dim: reduced dimension calculated successfully");
    return reduced_dim;
}

int get_num_reduction_batches(int* in_shape, int in_ndim, int reduced_dim)
{
    LOG_DEBUG("get_num_reduction_batches: Entering function with in_ndim=%d, reduced_dim=%d",
              in_ndim, reduced_dim);

    int num_batches = 1;
    for (int i = 0; i < in_ndim; ++i)
    {
        if (i != reduced_dim) num_batches *= in_shape[i];
    }

    LOG_DEBUG("get_num_reduction_batches: Number of batches: %d", num_batches);
    LOG_INFO("get_num_reduction_batches: number of reduction batches calculated successfully");
    return num_batches;
}

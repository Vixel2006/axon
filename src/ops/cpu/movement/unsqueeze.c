#include "ops/cpu/init.h" // For borrow
#include "ops/cpu/movement.h"

void unsqueeze_op(Tensor* in, Tensor* out, int dim)
{
    LOG_INFO("OP: unsqueeze_op: Unsqueezing Tensor %p at dimension %d", (void*) in, dim);

    if (dim < 0 || dim > in->ndim)
    {
        LOG_ERROR("unsqueeze_op: Invalid dimension %d for unsqueeze operation "
                  "(ndim=%d).",
                  dim, in->ndim);
        return;
    }

    for (int i = 0; i < dim; ++i)
    {
        out->strides[i] = in->strides[i];
    }
    for (int i = dim + 1; i < out->ndim; ++i)
    {
        out->strides[i] = in->strides[i - 1];
    }
    if (dim < out->ndim - 1)
    {
        out->strides[dim] = out->strides[dim + 1];
    }
    else
    {
        out->strides[dim] = out->strides[dim - 1];
    }

    borrow(out, in->data, in->grad);
}

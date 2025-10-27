#include "ops/cpu/init.h" // For borrow
#include "ops/cpu/movement.h"

void squeeze_op(Tensor* in, Tensor* out, int dim)
{
    LOG_INFO("OP: squeeze_op: Squeezing Tensor %p at dimension %d", (void*) in, dim);

    if (dim < 0 || dim >= in->ndim)
    {
        LOG_ERROR("squeeze_op: Invalid dimension %d for squeeze operation "
                  "(ndim=%d).",
                  dim, in->ndim);
        return;
    }

    if (in->shape[dim] != 1)
    {
        LOG_ERROR("Cannot squeeze dimension %d, size != 1", dim);
        return;
    }

    for (int i = 0; i < out->ndim; ++i)
        out->strides[i] = (i < dim) ? in->strides[i] : in->strides[i + 1];

    borrow(out, in->data, in->grad);
}

#include "ops/cpu/init.h" // For borrow
#include "ops/cpu/movement.h"

// WARNING: Assumes in->ndim == out->ndim
void expand_op(Tensor* in, Tensor* out, const int* shape)
{
    LOG_INFO("OP: expand_op: Expanding Tensor %p", (void*) in);

    for (int i = 0; i < in->ndim; ++i)
    {
        if (in->shape[i] != 1 && in->shape[i] != out->shape[i])
        {
            LOG_ERROR("expand_op: Cannot expand dimension %d from %d to %d. "
                      "Dimension must be 1 or match target size.",
                      i, in->shape[i], out->shape[i]);
            free(out->shape);
            free(out->strides);
            return;
        }
        out->strides[i] = (in->shape[i] == 1) ? 0 : in->strides[i];
    }
    borrow(out, in->data, in->grad);
}

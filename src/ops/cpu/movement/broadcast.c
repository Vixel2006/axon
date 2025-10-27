#include "ops/cpu/init.h" // For borrow
#include "ops/cpu/movement.h"

void broadcast_op(Tensor* in, Tensor* out, int ndim, const int* shape)
{
    if (!in || !out || !shape)
    {
        LOG_ERROR("broadcast_op ERROR: Input tensor, output tensor, or shape "
                  "array is NULL! in=%p, out=%p, shape=%p",
                  (void*) in, (void*) out, (void*) shape);
        return;
    }

    int in_dim = in->ndim - 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        if (in_dim >= 0)
        {
            if (in->shape[in_dim] == shape[i])
            {
                out->strides[i] = in->strides[in_dim];
            }
            else if (in->shape[in_dim] == 1)
            {
                out->strides[i] = 0;
            }
            else
            {
                LOG_ERROR("broadcast_op: Cannot broadcast dimension %d from %d to "
                          "%d. Dimension must be 1 or match target size.",
                          in_dim, in->shape[in_dim], shape[i]);
                free(out->shape);
                free(out->strides);
                return;
            }
            in_dim--;
        }
        else
        {
            out->strides[i] = 0;
        }
    }

    borrow(out, in->data, in->grad);
}

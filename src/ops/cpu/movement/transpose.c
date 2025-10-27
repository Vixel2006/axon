#include "ops/cpu/init.h" // For borrow
#include "ops/cpu/movement.h"

void transpose_op(Tensor* in, Tensor* out, int N, int M)
{
    if (!in || !out)
    {
        LOG_ERROR("transpose_op: Input or output tensor is NULL.");
        return;
    }

    LOG_INFO("OP: transpose_op: Transposing Tensor %p (dims %d, %d)", (void*) in, N, M);

    if (N < 0 || N >= in->ndim || M < 0 || M >= in->ndim || N == M)
    {
        LOG_ERROR("transpose_op: Invalid dimensions N=%d or M=%d for transpose operation "
                  "(ndim=%d). N and M must be within bounds and different.",
                  N, M, in->ndim);
        return;
    }

    for (int i = 0; i < out->ndim; ++i)
    {
        if (i == N)
        {
            out->strides[i] = in->strides[M];
        }
        else if (i == M)
        {
            out->strides[i] = in->strides[N];
        }
        else
        {
            out->strides[i] = in->strides[i];
        }
    }

    borrow(out, in->data, in->grad);
}

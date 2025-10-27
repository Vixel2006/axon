#include "ops/cpu/init.h"
#include "ops/cpu/movement.h"

void view_op(Tensor* in, Tensor* out, int* shape, int ndim)
{
    LOG_INFO("OP: view_op: Creating view from Tensor %p (ndim=%d)", (void*) in, ndim);
    borrow(out, in->data, in->grad);
}

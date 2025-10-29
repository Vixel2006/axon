#ifndef AUTOGRAD_CPU_BINARY_UTILS_H
#define AUTOGRAD_CPU_BINARY_UTILS_H

#include "autograd/cpu/binary/common.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif
    typedef float (*unary_grad_fn)(float dout, float aval, float scalar);
    typedef float (*binary_grad_fn)(float dout, float aval, float bval);

    void unary_grad_noncontig(Tensor* out, Tensor* a, float scalar, unary_grad_fn da_fn);
    void binary_grad_noncontig(Tensor* out, Tensor* a, Tensor* b, binary_grad_fn da_fn,
                               binary_grad_fn db_fn);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CPU_BINARY_UTILS_H

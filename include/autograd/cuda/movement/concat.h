#ifndef AUTOGRAD_CUDA_MOVEMENT_CONCAT_H
#define AUTOGRAD_CUDA_MOVEMENT_CONCAT_H

#include "axon_export.h"
#include "tensor.h"
#include "autograd/autograd_movement.h" // For ConcatExtras

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void concat_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);

#ifdef __cplusplus
}
#endif

#endif // AUTOGRAD_CUDA_MOVEMENT_CONCAT_H

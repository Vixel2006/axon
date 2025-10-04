#ifndef AXON_MOVEMENT_GRAD
#define AXON_MOVEMENT_GRAD

#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <stdlib.h>

#include "axon_export.h" // Include the generated export header

#ifdef __cplusplus
extern "C"
{
#endif
    AXON_EXPORT void concat_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

    AXON_EXPORT void concat_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
#ifdef __cplusplus
}
#endif

#endif

#ifndef AXON_INIT_OPS_H
#define AXON_INIT_OPS_H

#include "tensor.h"
#include <stdlib.h>
#include <time.h>

#include "axon_export.h" // Include the generated export header

#ifdef __cplusplus
extern "C"
{
#endif
    AXON_EXPORT void zeros_cpu(Tensor* t);
    AXON_EXPORT void ones_cpu(Tensor* t);
    AXON_EXPORT void randn_cpu(Tensor* t);
    AXON_EXPORT void uniform_cpu(Tensor* t, float low, float high);
    AXON_EXPORT void from_data_cpu(Tensor* t, float* data);
    AXON_EXPORT void borrow(Tensor* t, Storage* data, Storage* grad);

    AXON_EXPORT void zeros_cuda(Tensor* t);
    AXON_EXPORT void ones_cuda(Tensor* t);
    AXON_EXPORT void randn_cuda(Tensor* t);
    AXON_EXPORT void uniform_cuda(Tensor* t, float low, float high);
    AXON_EXPORT void from_data_cuda(Tensor* t, float* data);
#ifdef __cplusplus
}
#endif

#endif

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
    AXON_EXPORT void zeros(Tensor* t);
    AXON_EXPORT void ones(Tensor* t);
    AXON_EXPORT void randn(Tensor* t);
    AXON_EXPORT void uniform(Tensor* t, float low, float high);
    AXON_EXPORT void from_data(Tensor* t, float* data);
    AXON_EXPORT void borrow(Tensor* t, Storage* data, Storage* grad);
#ifdef __cplusplus
}
#endif

#endif

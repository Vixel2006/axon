#ifndef AXON_OPS_CUDA_INIT_H
#define AXON_OPS_CUDA_INIT_H

#include "axon_export.h"
#include "core_types.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void from_data_cuda(Tensor* t, float* data);

#ifdef __cplusplus
}
#endif

#endif // AXON_OPS_CUDA_INIT_H
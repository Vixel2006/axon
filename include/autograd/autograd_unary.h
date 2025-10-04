#ifndef AXON_UNARY_GRAD
#define AXON_UNARY_GRAD
#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#include "axon_export.h" // Include the generated export header

typedef struct
{
    float min_val;
    float max_val;
} ClipExtras;

#ifdef __cplusplus
extern "C"
{
#endif
    AXON_EXPORT void relu_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void abs_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void log_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void exp_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void neg_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void clip_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

    AXON_EXPORT void relu_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void abs_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void log_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void exp_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void neg_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void clip_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
#ifdef __cplusplus
}
#endif

#endif

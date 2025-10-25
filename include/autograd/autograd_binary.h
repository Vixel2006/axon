#ifndef AXON_BINARY_GRAD
#define AXON_BINARY_GRAD
#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#include "axon_export.h" // Include the generated export header

#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct
    {
        int padding;
        int H_in;
        int W_in;
        int Kh;
        int Kw;
        int Sh;
        int Sw;
        int Hout;
        int Wout;
    } BackwardConvExtras;

    typedef struct
    {
        int N;
        int K;
        int M;
    } MatMulBackwardExtras;

    AXON_EXPORT void add_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void sub_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void rsub_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void mul_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void div_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void rdiv_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void matmul_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void conv2d_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void dot_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void pow_grad_op_cpu(Tensor* out, Tensor** prev, int n_prev, void* extras);

    AXON_EXPORT void add_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void sub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void rsub_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void mul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void pow_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void div_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void rdiv_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void matmul_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void conv2d_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
    AXON_EXPORT void dot_grad_op_cuda(Tensor* out, Tensor** prev, int n_prev, void* extras);
#ifdef __cplusplus
}
#endif

#endif

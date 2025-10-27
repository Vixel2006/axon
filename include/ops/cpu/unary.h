#ifndef AXON_OPS_CPU_UNARY_H
#define AXON_OPS_CPU_UNARY_H

#include "logger.h"
#include "ops/init_ops.h"
#include "ops/unary_ops.h"
#include "utils.h"
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#if DEBUG
#include "utils.h"
#endif

#define EPS 1e-9
#define SAFE_LOGF(x) logf(((x) < EPS) ? EPS : (x))

#define SIMD_WIDTH 8

#define COMPUTE_UNARY_OFFSETS(linear_idx, in_tensor, off_in, off_out)                              \
    do                                                                                             \
    {                                                                                              \
        int idx = linear_idx;                                                                      \
        off_in = 0;                                                                                \
        off_out = 0;                                                                               \
        for (int d = in_tensor->ndim - 1; d >= 0; --d)                                             \
        {                                                                                          \
            int coord = idx % in_tensor->shape[d];                                                 \
            idx /= in_tensor->shape[d];                                                            \
            off_in += coord * in_tensor->strides[d];                                               \
            off_out += coord * out->strides[d];                                                    \
        }                                                                                          \
    } while (0)

static inline bool check_tensors_unary(Tensor* in, Tensor* out, const char* op_name)
{
    if (!in || !out)
    {
        LOG_ERROR("%s ERROR: NULL tensor! in=%p, out=%p", op_name, (void*) in, (void*) out);
        return false;
    }
    return true;
}

static inline float* alloc_tensor_data(int size, const char* op_name)
{
    float* data = (float*) malloc(sizeof(float) * size);
    if (!data)
    {
        LOG_ERROR("%s ERROR: Failed to allocate memory for %d floats", op_name, size);
        return NULL;
    }
    return data;
}

static inline bool can_use_simd_unary(Tensor* in, Tensor* out)
{
    return is_contiguous(in) && is_contiguous(out);
}

#endif // AXON_OPS_CPU_UNARY_H

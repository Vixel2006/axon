#ifndef AXON_OPS_CPU_BINARY_H
#define AXON_OPS_CPU_BINARY_H

#include "logger.h"
#include "ops/binary_ops.h"
#include "ops/init_ops.h"
#include "tensor.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

extern __m256 Sleef_powf8_u10(__m256 x, __m256 y);

#define SIMD_WIDTH 8

#define COMPUTE_OFFSETS(linear_idx, tensor_for_shape, str_a, str_b, str_out, off_a, off_b,         \
                        off_out)                                                                   \
    do                                                                                             \
    {                                                                                              \
        int idx = linear_idx;                                                                      \
        off_a = 0;                                                                                 \
        off_b = 0;                                                                                 \
        off_out = 0;                                                                               \
        for (int d = tensor_for_shape->ndim - 1; d >= 0; --d)                                      \
        {                                                                                          \
            int coord = idx % tensor_for_shape->shape[d];                                          \
            idx /= tensor_for_shape->shape[d];                                                     \
            off_a += coord * str_a[d];                                                             \
            off_b += coord * str_b[d];                                                             \
            off_out += coord * str_out[d];                                                         \
        }                                                                                          \
    } while (0)

static inline bool check_tensors(Tensor* a, Tensor* b, Tensor* out, const char* op_name)
{
    if (!a || !b || !out)
    {
        LOG_ERROR("%s ERROR: NULL tensor! a=%p, b=%p, out=%p", op_name, (void*) a, (void*) b,
                  (void*) out);
        return false;
    }
    return true;
}

static inline bool check_tensors_unary_or_dot(Tensor* a, Tensor* out, const char* op_name)
{
    if (!a || !out)
    {
        LOG_ERROR("%s ERROR: NULL tensor! a=%p, out=%p", op_name, (void*) a, (void*) out);
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

static inline bool can_use_simd(Tensor* a, Tensor* b, Tensor* out)
{
    return is_contiguous(a) && is_contiguous(b) && is_contiguous(out);
}

#endif // AXON_OPS_CPU_BINARY_H

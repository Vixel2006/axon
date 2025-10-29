#ifndef AUTOGRAD_CPU_BINARY_COMMON_H
#define AUTOGRAD_CPU_BINARY_COMMON_H

#include "autograd/autograd_binary.h"
#include "autograd/autograd_utils.h" // For numel, is_contiguous, shapes_equal
#include "autograd/cpu/binary/utils.h"
#include "logger.h"
#include "tensor.h"
#include "utils.h" // For numel, is_contiguous, shapes_equal
#include <assert.h>
#include <immintrin.h> // For SIMD operations
#include <math.h>      // For powf, logf
#include <stdlib.h>    // For NULL

#define SIMD_WIDTH 8

typedef float (*unary_grad_fn)(float dout, float aval, float scalar);
typedef float (*binary_grad_fn)(float dout, float aval, float bval);

static inline float binary_add_da(float dout, float a, float b) { return dout; }
static inline float binary_add_db(float dout, float a, float b) { return dout; }
static inline float binary_sub_da(float dout, float a, float b) { return dout; }
static inline float binary_sub_db(float dout, float a, float b) { return -dout; }
static inline float binary_mul_da(float dout, float a, float b) { return dout * b; }
static inline float binary_mul_db(float dout, float a, float b) { return dout * a; }
static inline float binary_div_da(float dout, float a, float b)
{
    return (b != 0.0f) ? dout / b : 0.0f;
}
static inline float binary_div_db(float dout, float a, float b)
{
    return (b != 0.0f) ? -dout * a / (b * b) : 0.0f;
}

static inline float binary_pow_da(float dout, float a, float b)
{
    float grad_val = 0.0f;
    if (!(a == 0.0f && (b - 1.0f) < 0.0f))
    {
        grad_val = b * powf(a, b - 1.0f);
    }
    return dout * grad_val;
}
static inline float binary_pow_db(float dout, float a, float b)
{
    float grad_val = 0.0f;
    if (a > 0.0f)
    { // log is only defined for positive numbers
        grad_val = powf(a, b) * logf(a);
    }
    return dout * grad_val;
}

static inline float unary_add_da(float dout, float a, float scalar) { return dout; }
static inline float unary_sub_da(float dout, float a, float scalar) { return dout; }
static inline float unary_rsub_da(float dout, float a, float scalar) { return -dout; }
static inline float unary_mul_da(float dout, float a, float scalar) { return dout * scalar; }
static inline float unary_div_da(float dout, float a, float scalar)
{
    return (scalar != 0.0f) ? dout / scalar : 0.0f;
}
static inline float unary_rdiv_da(float dout, float a, float scalar)
{
    return (a != 0.0f) ? -scalar / (a * a) * dout : 0.0f;
}
static inline float unary_pow_da(float dout, float a, float scalar)
{
    float grad_val = 0.0f;
    if (!(a == 0.0f && (scalar - 1.0f) < 0.0f))
    {
        grad_val = scalar * powf(a, scalar - 1.0f);
    }
    return dout * grad_val;
}

#endif // AUTOGRAD_CPU_BINARY_COMMON_H

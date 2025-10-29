#ifndef AUTOGRAD_CPU_UNARY_COMMON_H
#define AUTOGRAD_CPU_UNARY_COMMON_H

#include "autograd/autograd_unary.h"
#include "autograd/autograd_utils.h" // For numel, is_contiguous, shapes_equal
#include "logger.h"
#include "tensor.h"
#include "utils.h" // For numel, is_contiguous, shapes_equal
#include <assert.h>
#include <immintrin.h> // For SIMD operations
#include <math.h>      // For fabsf

#define SIMD_WIDTH 8

#endif // AUTOGRAD_CPU_UNARY_COMMON_H

#ifndef AXON_OPS_CPU_REDUCTION_H
#define AXON_OPS_CPU_REDUCTION_H

#include "utils.h"
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "ops/reduction_ops.h"
#include "ops/cpu/init.h" // For from_data

#define SIMD_WIDTH 8

void sum_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim);
void mean_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim);
void max_op_cpu(Tensor* a, Tensor* out, int axis, bool keepdim);
void sum_full_op_cpu(Tensor* a, Tensor* out);
void mean_full_op_cpu(Tensor* a, Tensor* out);
void max_full_op_cpu(Tensor* a, Tensor* out);

#endif // AXON_OPS_CPU_REDUCTION_H

#ifndef AXON_OPS_CPU_MOVEMENT_H
#define AXON_OPS_CPU_MOVEMENT_H

#include "utils.h"
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "ops/movement_ops.h"

#define MAX_NDIM 32

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

#endif // AXON_OPS_CPU_MOVEMENT_H

#ifndef AXON_MOVEMENT_OPS_H
#define AXON_MOVEMENT_OPS_H

#include "init_ops.h"
#include "tensor.h"

#include "ops/cpu/movement.h"

#ifdef __CUDACC__
#include "ops/cuda/movement.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

// Individual function declarations are now in ops/cpu/movement.h and ops/cuda/movement.h

#ifdef __cplusplus
}
#endif

#endif

#ifndef AXON_MOVEMENT_GRAD
#define AXON_MOVEMENT_GRAD

#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <stdlib.h>

#include "axon_export.h" // Include the generated export header

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int axis;
    } ConcatExtras;

    // CPU Movement Ops
    #include "autograd/cpu/movement/concat.h"

    // CUDA Movement Ops
    #include "autograd/cuda/movement/concat.h"

#ifdef __cplusplus
}
#endif

#endif

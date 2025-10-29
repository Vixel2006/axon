#ifndef AXON_REDUCTION_GRAD
#define AXON_REDUCTION_GRAD

#include "utils.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include "autograd_utils.h"
#include "axon_export.h" // Include the generated export header
#include "logger.h"
#include "ops/init_ops.h"

#ifdef __cplusplus
extern "C"
{
#endif
    // CPU Reduction Ops
    #include "autograd/cpu/reduction/sum.h"
    #include "autograd/cpu/reduction/mean.h"
    #include "autograd/cpu/reduction/max.h"
    #include "autograd/cpu/reduction/sum_full.h"
    #include "autograd/cpu/reduction/mean_full.h"
    #include "autograd/cpu/reduction/max_full.h"

    // CUDA Reduction Ops
    #include "autograd/cuda/reduction/sum.h"
    #include "autograd/cuda/reduction/mean.h"
    #include "autograd/cuda/reduction/max.h"
    #include "autograd/cuda/reduction/sum_full.h"
    #include "autograd/cuda/reduction/mean_full.h"
    #include "autograd/cuda/reduction/max_full.h"
#ifdef __cplusplus
}
#endif

#endif

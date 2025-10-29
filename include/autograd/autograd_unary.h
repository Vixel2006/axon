#ifndef AXON_UNARY_GRAD
#define AXON_UNARY_GRAD
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
    // CPU Unary Ops
    #include "autograd/cpu/unary/relu.h"
    #include "autograd/cpu/unary/abs.h"
    #include "autograd/cpu/unary/log.h"
    #include "autograd/cpu/unary/exp.h"
    #include "autograd/cpu/unary/neg.h"
    #include "autograd/cpu/unary/clip.h"

    // CUDA Unary Ops
    #include "autograd/cuda/unary/relu.h"
    #include "autograd/cuda/unary/abs.h"
    #include "autograd/cuda/unary/log.h"
    #include "autograd/cuda/unary/exp.h"
    #include "autograd/cuda/unary/neg.h"
    #include "autograd/cuda/unary/clip.h"
#ifdef __cplusplus
}
#endif

#endif

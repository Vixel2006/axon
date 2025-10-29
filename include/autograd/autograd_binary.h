#ifndef AXON_BINARY_GRAD
#define AXON_BINARY_GRAD
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
    typedef struct
    {
        int padding;
        int H_in;
        int W_in;
        int Kh;
        int Kw;
        int Sh;
        int Sw;
        int Hout;
        int Wout;
    } BackwardConvExtras;

    typedef struct
    {
        int N;
        int K;
        int M;
    } MatMulBackwardExtras;

    // CPU Binary Ops
    #include "autograd/cpu/binary/add.h"
    #include "autograd/cpu/binary/sub.h"
    #include "autograd/cpu/binary/rsub.h"
    #include "autograd/cpu/binary/mul.h"
    #include "autograd/cpu/binary/div.h"
    #include "autograd/cpu/binary/rdiv.h"
    #include "autograd/cpu/binary/matmul.h"
    #include "autograd/cpu/binary/conv2d.h"
    #include "autograd/cpu/binary/dot.h"
    #include "autograd/cpu/binary/pow.h"

    // CUDA Binary Ops
    #include "autograd/cuda/binary/add.h"
    #include "autograd/cuda/binary/sub.h"
    #include "autograd/cuda/binary/rsub.h"
    #include "autograd/cuda/binary/mul.h"
    #include "autograd/cuda/binary/pow.h"
    #include "autograd/cuda/binary/div.h"
    #include "autograd/cuda/binary/rdiv.h"
    #include "autograd/cuda/binary/matmul.h"
    #include "autograd/cuda/binary/conv2d.h"
    #include "autograd/cuda/binary/dot.h"
#ifdef __cplusplus
}
#endif

#endif

#ifndef AXON_MOVEMENT_GRAD
#define AXON_MOVEMENT_GRAD

#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <stdlib.h>

void concat_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);

#endif

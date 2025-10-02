#ifndef IDRAK_MOVEMENT_GRAD
#define IDRAK_MOVEMENT_GRAD

#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <stdlib.h>

void concat_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);

#endif

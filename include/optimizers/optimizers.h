#ifndef NAWAH_OPTIMIZERS_H
#define NAWAH_OPTIMIZERS_H

#include "tensor.h"

void sgd(Tensor **params, int num_params, float lr);

#endif

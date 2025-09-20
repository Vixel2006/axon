#ifndef IDRAK_INIT_OPS_H
#define IDRAK_INIT_OPS_H

#include "tensor.h"
#include <stdlib.h>
#include <time.h>

void zeros(Tensor* t);
void ones(Tensor* t);
void randn(Tensor* t);
void uniform(Tensor* t, float low, float high);
void from_data(Tensor* t, float* data);
void borrow(Tensor* t, Storage* data);

#endif

#ifndef AXON_OPS_CPU_INIT_H
#define AXON_OPS_CPU_INIT_H

#include "logger.h"
#include "tensor.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

void zeros(Tensor* t);
void ones(Tensor* t);
void randn(Tensor* t);
void uniform(Tensor* t, float low, float high);
void from_data(Tensor* t, float* data);
void borrow(Tensor* t, Storage* data, Tensor* grad);

#endif // AXON_OPS_CPU_INIT_H

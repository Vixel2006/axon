#ifndef AXON_UTILS_H
#define AXON_UTILS_H

#include "logger.h"
#include "tensor.h"

void print_shape(const int* shape, int ndim);
int get_flat_index(const Tensor* t, const int* indices);

#endif

#ifndef IDRAK_UTILS_H
#define IDRAK_UTILS_H

#include "logger.h"
#include "tensor.h" // Required for Tensor type in get_flat_index

void print_shape(const int *shape, int ndim);
int get_flat_index(const Tensor *t, const int *indices);

#endif



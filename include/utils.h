#ifndef NAWAH_UTILS_H
#define NAWAH_UTILS_H

#include "tensor.h"

int get_num_batches(const int *shape, int ndim);
int get_flat_index(const Tensor *t, const int *indices);

#endif

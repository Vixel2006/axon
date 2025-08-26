#ifndef NAWAH_TENSOR_H
#define NAWAH_TENSOR_H

#include <stdlib.h>

typedef struct {
  float *data;
  int ndim;
  int *shape;
  int *strides;
  bool requires_grad;
  float *grad;
  void *grad_fn;
} Tensor;

Tensor *malloc_tensor_empty();
Tensor *malloc_tensor_shape(const int *shape, int ndim, bool requires_grad);
Tensor *malloc_tensor_full(const int *shape, int ndim, const int *strides,
                           float *data, bool requires_grad, float *grad);

void free_tensor(Tensor *t);

int numel(const int *shape, int ndim);
int *compute_strides(const int *shape, int ndim);

#endif

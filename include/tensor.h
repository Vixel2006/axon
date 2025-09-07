#ifndef IDRAK_TENSOR_H
#define IDRAK_TENSOR_H

#include <stdbool.h>
#include <stdlib.h>

typedef struct {
  float *ptr;
  unsigned int ref_counter;
} SharedPtr;

typedef struct {
  SharedPtr *data;
  SharedPtr *grad;
  int ndim;
  int *shape;
  int *strides;
  bool requires_grad;
  void *grad_fn;
} Tensor;

SharedPtr *malloc_shared_ptr(float *ptr, int size);
void free_shared_ptr(SharedPtr **ptr);

Tensor *malloc_tensor_empty();
Tensor *malloc_tensor_shape(const int *shape, int ndim, bool requires_grad);
Tensor *malloc_tensor_full(const int *shape, int ndim, const int *strides,
                           SharedPtr *data, bool requires_grad,
                           SharedPtr *grad);

void free_tensor(Tensor **t);

int numel(const int *shape, int ndim);
int *compute_strides(const int *shape, int ndim);
void set_ones_grad(Tensor *t);
bool is_contiguous(Tensor *t);

Tensor *zeros(const int *shape, int ndim, bool requires_grad);
Tensor *ones(const int *shape, int ndim, bool requires_grad);
Tensor *randn(const int *shape, int ndim, int seed, bool requires_grad);
Tensor *uniform(const int *shape, int ndim, float low, float high,
                bool requires_grad);

#endif

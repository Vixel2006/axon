#ifndef IDRAK_TENSOR_H
#define IDRAK_TENSOR_H

#include "shared_ptr.h"
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    shared_ptr* data;
    shared_ptr* grad;
    int* shape;
    int* strides;
    int ndim;
    bool requires_grad;
} Tensor;

Tensor* tmalloc_shape(const int* shape, int ndim, Dtype dtype, Device device, bool requires_grad);
Tensor* tmalloc_full(const int* shape, int ndim, const int* strides, Dtype dtype, Device device, shared_ptr* data, bool requires_grad, shared_ptr* grad);
void tfree(Tensor* t);

int numel(const int* shape, int ndim);
int* compute_strides(const int* shape, int ndim);
void set_ones_grad(Tensor* t);
bool is_contiguous(Tensor* t);

Tensor* zeros(const int* shape, int ndim, Dtype dtype, Device device, bool requires_grad);
Tensor* ones(const int* shape, int ndim, Dtype dtype, Device device, bool requires_grad);
Tensor* randn(const int* shape, int ndim, Dtype dtype, Device device, int seed, bool requires_grad);
Tensor* uniform(const int* shape, int ndim, Dtype dtype, Device device, float low, float high, bool requires_grad);

#endif

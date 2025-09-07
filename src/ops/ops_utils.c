#include "ops/ops_utils.h"
#include <stdio.h>

int tensor_alloc_shape(int ndim, const int *shape, int **out_shape) {
  if (!*out_shape)
    return -1;
  memcpy(*out_shape, shape, ndim * sizeof(int));
  return 0;
}

int tensor_alloc_strides(int ndim, const int *strides, int **out_strides) {
  if (!*out_strides)
    return -1;
  memcpy(*out_strides, strides, ndim * sizeof(int));
  return 0;
}

int tensor_copy_layout(Tensor *in, Tensor *out, const int *shape) {
  out->ndim = in->ndim;
  if (tensor_alloc_shape(out->ndim, shape, &out->shape) != 0)
    return -1;
  out->strides = NULL;
  return 0;
}

void reference_shared_ptr(SharedPtr **out, SharedPtr *in) {
  if (!in) {
    *out = NULL;
    return;
  }

  *out = in;
  in->ref_counter++;
}

void tensor_init_view(Tensor *out, Tensor *in) {
  // Input validation
  if (!out || !in) {
    return;
  }

  reference_shared_ptr(&out->data, in->data);
  reference_shared_ptr(&out->grad, in->grad);

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

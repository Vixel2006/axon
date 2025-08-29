#include "ops/ops_utils.h"

int tensor_alloc_shape(int ndim, const int *shape, int **out_shape) {
  *out_shape = malloc(ndim * sizeof(int));
  if (!*out_shape) return -1;
  memcpy(*out_shape, shape, ndim * sizeof(int));
  return 0;
}

// Allocate and copy strides
int tensor_alloc_strides(int ndim, const int *strides, int **out_strides) {
  *out_strides = malloc(ndim * sizeof(int));
  if (!*out_strides) return -1;
  memcpy(*out_strides, strides, ndim * sizeof(int));
  return 0;
}

// Copy shape+strides from in → out, but replace shape with `shape`
int tensor_copy_layout(Tensor *in, Tensor *out, const int *shape) {
  out->ndim = in->ndim;
  if (tensor_alloc_shape(out->ndim, shape, &out->shape) != 0) return -1;
  out->strides = NULL; // Strides will be computed by the caller
  return 0;
}

// Initialize out tensor as a "view" (share data/grad, no copy)
void tensor_init_view(Tensor *out, Tensor *in) {
  out->owns_data = false;
  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

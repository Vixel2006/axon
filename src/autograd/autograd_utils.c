#include "autograd/autograd_utils.h"
#include "utils.h"

int get_reduced_dim(int *in_shape, int *out_shape, int in_ndim, int out_ndim) {
  DEBUG_PRINT("get_reduced_dim: Calculating reduced dimension\n");

  int reduced_dim = -1;
  for (int i = 0; i < in_ndim; ++i) {
    if (out_ndim <= i || in_shape[i] != out_shape[i]) {
      reduced_dim = i;
      break;
    }
  }

  return reduced_dim;
}

int get_num_reduction_batches(int *in_shape, int in_ndim, int reduced_dim) {
  DEBUG_PRINT(
      "get_num_reduction_batches: Calculating number of reduction batches\n");

  int num_batches = 1;
  for (int i = 0; i < in_ndim; ++i) {
    if (i != reduced_dim)
      num_batches *= in_shape[i];
  }
  return num_batches;
}

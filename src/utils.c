#include "utils.h"

int get_num_batches(const int *shape, int ndim) {
  int batch_nums = 1;
  for (int i = 0; i < ndim - 2; ++i) {
    batch_nums *= shape[i];
  }

  return batch_nums;
}

int get_flat_index(const Tensor *t, const int *indices) {
  int flat_index = 0;
  for (int i = 0; i < t->ndim; ++i) {
    flat_index += indices[i] * t->strides[i];
  }
  return flat_index;
}

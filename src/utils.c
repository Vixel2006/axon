#include "utils.h"

int get_num_batches(const int *shape, int ndim) {
  int batch_nums = 1;
  for (int i = 0; i < ndim - 2; ++i) {
    batch_nums *= shape[i];
  }

  return batch_nums;
}
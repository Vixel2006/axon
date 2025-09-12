#include "utils.h"
#include "logger.h"
#include "tensor.h" // Added this line
#include <stdio.h>

int _idrak_debug_enabled = 0;

void idrak_set_debug_mode(int enable) {
    _idrak_debug_enabled = enable;
    LOG_INFO("Debug mode set to %d", enable);
}

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

void print_shape(const int *shape, int ndim) {
    if (!_idrak_debug_enabled) {
        return;
    }
    LOG_INFO("[Shape] (");
    for (int i = 0; i < ndim; ++i) {
        fprintf(stderr, "%d", shape[i]);
        if (i < ndim - 1) {
            fprintf(stderr, ", ");
        }
    }
    fprintf(stderr, ")\n");
}

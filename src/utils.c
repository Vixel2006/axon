#include "utils.h"

int _idrak_debug_enabled = 0;

#include <stdio.h>
#include "utils.h"

void idrak_set_debug_mode(int enable) {
    _idrak_debug_enabled = enable;
    DEBUG_PRINT(ANSI_COLOR_CYAN, "Debug mode set to %d\n", enable);
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
    fprintf(stderr, ANSI_COLOR_CYAN "[DEBUG] [Shape] (" ANSI_COLOR_RESET);
    for (int i = 0; i < ndim; ++i) {
        fprintf(stderr, "%d", shape[i]);
        if (i < ndim - 1) {
            fprintf(stderr, ", ");
        }
    }
    fprintf(stderr, ANSI_COLOR_CYAN ")\n" ANSI_COLOR_RESET);
}


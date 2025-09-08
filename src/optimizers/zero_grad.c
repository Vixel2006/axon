#include "optimizers/optimizers.h"
#include "utils.h"
#include <string.h>

void zero_grad(Tensor **parameters, int num_parameters) {
  DEBUG_PRINT("[IDRAK_DEBUG] zero_grad: Zeroing gradients for %d parameters\n",
              num_parameters);

  for (int i = 0; i < num_parameters; ++i) {
    Tensor *t = parameters[i];

    // 1. Check if the Tensor pointer itself is valid
    if (t == NULL) {
      fprintf(stderr,
              "Warning: Parameter at index %d is NULL, skipping zero_grad.\n",
              i);
      continue;
    }

    // 2. Check if gradients are required for this tensor
    if (!t->requires_grad) {
      // fprintf(stderr, "Info: Parameter at index %d does not require
      // gradients, skipping zero_grad.\n", i);
      continue;
    }

    // 3. Check if the SharedData struct for gradients is allocated
    if (t->grad == NULL) {
      fprintf(stderr,
              "Warning: Parameter at index %d requires gradients but t->grad "
              "(SharedData) is NULL. Cannot zero gradient.\n",
              i);
      continue;
    }

    // 4. Check if the actual gradient data pointer (float array) is allocated
    if (t->grad->ptr == NULL) {
      fprintf(stderr,
              "Warning: Parameter at index %d requires gradients but "
              "t->grad->ptr is NULL. Cannot zero gradient.\n",
              i);
      continue;
    }

    // All checks passed, proceed to zero the gradient buffer
    size_t size = numel(t->shape, t->ndim);
    if (size > 0) { // Only memset if there are elements
      memset(t->grad->ptr, 0, size * sizeof(float));
      DEBUG_PRINT("[IDRAK_DEBUG] zero_grad: Zeroed gradient for parameter %d "
                  "(size=%zu)\n",
                  i, size);
    }
  }
}

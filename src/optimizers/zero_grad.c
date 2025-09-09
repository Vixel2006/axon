#include "optimizers/optimizers.h"
#include "utils.h"
#include <string.h>

void zero_grad(Tensor **parameters, int num_parameters) {
  IDRAK_DEBUG("DEBUG", "zero_grad: Zeroing gradients for %d parameters\n",
              num_parameters);

  for (int i = 0; i < num_parameters; ++i) {
    Tensor *t = parameters[i];

    if (t == NULL) {
      IDRAK_WARNING(
          "zero_grad: Parameter at index %d is NULL, skipping zero_grad.\n", i);
      continue;
    }

    if (!t->requires_grad) {
      continue;
    }

    if (t->grad == NULL) {
      IDRAK_WARNING("zero_grad: Parameter at index %d requires gradients but "
                    "t->grad (SharedData) is NULL. Cannot zero gradient.\n",
                    i);
      continue;
    }

    if (t->grad->ptr == NULL) {
      IDRAK_WARNING("zero_grad: Parameter at index %d requires gradients but "
                    "t->grad->ptr is NULL. Cannot zero gradient.\n",
                    i);
      continue;
    }

    size_t size = numel(t->shape, t->ndim);
    if (size > 0) {
      memset(t->grad->ptr, 0, size * sizeof(float));
      IDRAK_DEBUG("DEBUG",
                  "zero_grad: Zeroed gradient for parameter %d (size=%zu)\n", i,
                  size);
    }
  }
}

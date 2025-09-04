#include "optimizers/optimizers.h"
#include <string.h>

void zero_grad(Tensor **parameters, int num_parameters) {
  for (int i = 0; i < num_parameters; ++i) {
    Tensor *t = parameters[i];
    if (t->grad != NULL) {
      int size = numel(t->shape, t->ndim);
      memset(t->grad, 0, size * sizeof(float));
    }
  }
}

#include "tensor.h"
#include "tape.h"
#include <stdlib.h>
#include <string.h>

int numel(const int *shape, int ndim) {
  if (ndim <= 0 || !shape)
    return 0;
  int size = 1;
  for (int i = 0; i < ndim; ++i) {
    if (shape[i] <= 0)
      return 0;
    size *= shape[i];
  }
  return size;
}

int *compute_strides(const int *shape, int ndim) {
  if (ndim <= 0 || !shape)
    return NULL;

  int *strides = malloc(ndim * sizeof(int));
  if (!strides)
    return NULL;

  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

Tensor *malloc_tensor_empty() {
  Tensor *t = malloc(sizeof(Tensor));
  if (!t)
    return NULL;

  memset(t, 0, sizeof(Tensor));
  return t;
}

Tensor *malloc_tensor_shape(const int *shape, int ndim, bool requires_grad) {
  if (ndim < 0 || (ndim > 0 && !shape))
    return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t)
    return NULL;

  memset(t, 0, sizeof(Tensor));
  t->ndim = ndim;

  if (ndim == 0) {
    t->shape = NULL;
    t->strides = NULL;
    t->data = malloc(sizeof(float));
    if (!t->data) {
      free(t);
      return NULL;
    }
    t->data[0] = 0.0f;

    if (requires_grad) {
      t->requires_grad = true;
      t->grad = malloc(sizeof(float));
      if (!t->grad) {
        free(t->data);
        free(t);
        return NULL;
      }
      t->grad[0] = 0.0f;

      t->ctx = malloc(sizeof(Tape));
      if (!t->ctx) {
        free(t->grad);
        free(t->data);
        free(t);
        return NULL;
      }
      memset(t->ctx, 0, sizeof(Tape));
    }
    return t;
  }

  t->shape = malloc(ndim * sizeof(int));
  if (!t->shape) {
    free(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  int size = numel(shape, ndim);
  if (size <= 0) {
    free(t->shape);
    free(t);
    return NULL;
  }

  t->strides = compute_strides(t->shape, ndim);
  if (!t->strides) {
    free(t->shape);
    free(t);
    return NULL;
  }

  t->data = malloc(size * sizeof(float));
  if (!t->data) {
    free(t->strides);
    free(t->shape);
    free(t);
    return NULL;
  }

  memset(t->data, 0, size * sizeof(float));

  if (requires_grad) {
    t->requires_grad = true;
    t->grad = malloc(size * sizeof(float));
    if (!t->grad) {
      free(t->data);
      free(t->strides);
      free(t->shape);
      free(t);
      return NULL;
    }
    memset(t->grad, 0, size * sizeof(float));

    t->ctx = malloc(sizeof(Tape));
    if (!t->ctx) {
      free(t->grad);
      free(t->data);
      free(t->strides);
      free(t->shape);
      free(t);
      return NULL;
    }
    memset(t->ctx, 0, sizeof(Tape));
  }

  return t;
}

Tensor *malloc_tensor_full(const int *shape, int ndim, const int *strides,
                           float *data, bool requires_grad, float *grad) {
  if (ndim < 0 || (ndim > 0 && (!shape || !data)))
    return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t)
    return NULL;

  memset(t, 0, sizeof(Tensor));
  t->ndim = ndim;
  t->requires_grad = requires_grad;

  int size = 1;
  if (ndim == 0) {
    size = 1;
    t->shape = NULL;
    t->strides = NULL;
  } else {
    size = numel(shape, ndim);
    if (size <= 0) {
      free(t);
      return NULL;
    }

    t->shape = malloc(ndim * sizeof(int));
    if (!t->shape) {
      free(t);
      return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    t->strides = malloc(ndim * sizeof(int));
    if (!t->strides) {
      free(t->shape);
      free(t);
      return NULL;
    }
    if (strides) {
      memcpy(t->strides, strides, ndim * sizeof(int));
    } else {
      int *default_strides = compute_strides(shape, ndim);
      if (!default_strides) {
        free(t->strides);
        free(t->shape);
        free(t);
        return NULL;
      }
      memcpy(t->strides, default_strides, ndim * sizeof(int));
      free(default_strides);
    }
  }

  t->data = malloc(size * sizeof(float));
  if (!t->data) {
    free(t->strides);
    free(t->shape);
    free(t);
    return NULL;
  }
  memcpy(t->data, data, size * sizeof(float));

  if (requires_grad) {
    t->grad = malloc(size * sizeof(float));
    if (!t->grad) {
      free(t->data);
      free(t->strides);
      free(t->shape);
      free(t);
      return NULL;
    }

    if (grad) {
      memcpy(t->grad, grad, size * sizeof(float));
    } else {
      memset(t->grad, 0, size * sizeof(float));
    }

    t->ctx = malloc(sizeof(Tape));
    if (!t->ctx) {
      free(t->grad);
      free(t->data);
      free(t->strides);
      free(t->shape);
      free(t);
      return NULL;
    }
    memset(t->ctx, 0, sizeof(Tape));
  }

  return t;
}

void free_tensor(Tensor *t) {
  if (t) {
    free(t->data);
    free(t->grad);
    free(t->ctx);
    free(t->shape);
    free(t->strides);
    free(t);
  }
}

#include "tensor.h"

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool is_contiguous(Tensor *t) {
  int expected_stride = 1;

  for (int i = t->ndim - 1; i >= 0; --i) {
    if (t->shape[i] > 1) {
      if (t->strides[i] != expected_stride)
        return false;

      expected_stride *= t->shape[i];
    }
  }

  return true;
}

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

void set_ones_grad(Tensor *t) {
  int size = numel(t->shape, t->ndim);
  for (int i = 0; i < size; ++i) {
    t->grad[i] = 1.0f;
  }
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

  t->ndim = ndim;

  if (ndim == 0) {
    t->shape = NULL;
    t->strides = NULL;
    t->data = malloc(sizeof(float));
    if (!t->data) {
      free_tensor(t);
      return NULL;
    }

    t->data[0] = 0.0f;

    if (requires_grad) {
      t->requires_grad = true;
      t->grad = malloc(sizeof(float));
      if (!t->grad) {
        free_tensor(t);
        return NULL;
      }
      t->grad[0] = 0.0f;
    }
    return t;
  }

  t->shape = malloc(ndim * sizeof(int));
  if (!t->shape) {
    free_tensor(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  int size = numel(shape, ndim);
  if (size <= 0) {
    free_tensor(t);
    return NULL;
  }

  t->strides = compute_strides(t->shape, ndim);
  if (!t->strides) {
    free_tensor(t);
    return NULL;
  }

  t->data = malloc(size * sizeof(float));
  if (!t->data) {
    free_tensor(t);
    return NULL;
  }

  memset(t->data, 0, size * sizeof(float));

  t->owns_data = true;

  if (requires_grad) {
    t->requires_grad = true;
    t->grad = malloc(size * sizeof(float));
    if (!t->grad) {
      free_tensor(t);
      return NULL;
    }
    memset(t->grad, 0, size * sizeof(float));
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
      free_tensor(t);
      return NULL;
    }

    t->shape = malloc(ndim * sizeof(int));
    if (!t->shape) {
      free_tensor(t);
      return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    t->strides = malloc(ndim * sizeof(int));
    if (!t->strides) {
      free_tensor(t);
      return NULL;
    }
    if (strides) {
      memcpy(t->strides, strides, ndim * sizeof(int));
    } else {
      int *default_strides = compute_strides(shape, ndim);
      if (!default_strides) {
        free_tensor(t);
        return NULL;
      }
      memcpy(t->strides, default_strides, ndim * sizeof(int));
      free(default_strides);
    }
  }

  t->owns_data = true;

  t->data = malloc(size * sizeof(float));
  if (!t->data) {
    free_tensor(t);
    return NULL;
  }
  memcpy(t->data, data, size * sizeof(float));

  if (requires_grad) {
    t->grad = malloc(size * sizeof(float));
    if (!t->grad) {
      free_tensor(t);
      return NULL;
    }

    if (grad) {
      memcpy(t->grad, grad, size * sizeof(float));
    } else {
      memset(t->grad, 0, size * sizeof(float));
    }
  }

  return t;
}

void free_tensor(Tensor *t) {
  if (t) {
    if (t->data && t->owns_data)
      free(t->data);

    if (t->grad && t->owns_data)
      free(t->grad);

    if (t->shape)
      free(t->shape);

    if (t->strides)
      free(t->strides);

    free(t);
  }
}

Tensor *zeros(const int *shape, int ndim, bool requires_grad) {
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t)
    return NULL;

  // malloc_tensor_shape already sets data to 0, so nothing more to do here.
  return t;
}

Tensor *ones(const int *shape, int ndim, bool requires_grad) {
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t)
    return NULL;

  int size = numel(shape, ndim);
  for (int i = 0; i < size; ++i) {
    t->data[i] = 1.0f;
  }
  return t;
}

Tensor *uniform(const int *shape, int ndim, float low, float high,
                bool requires_grad) {
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t)
    return NULL;

  // Seed the random number generator if not already seeded.
  // This is a simple approach; in a real application, seeding should be done
  // once.
  static bool seeded = false;
  if (!seeded) {
    srand(time(NULL));
    seeded = true;
  }

  int size = numel(shape, ndim);
  for (int i = 0; i < size; ++i) {
    t->data[i] = low + (float)rand() / (RAND_MAX / (high - low));
  }
  return t;
}

Tensor *randn(const int *shape, int ndim, int seed, bool requires_grad) {
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t)
    return NULL;

  srand(seed); // Seed for reproducibility

  int size = numel(shape, ndim);
  for (int i = 0; i < size; i += 2) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;

    float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * M_PI * u2);

    t->data[i] = z1;
    if (i + 1 < size) {
      t->data[i + 1] = z2;
    }
  }
  return t;
}

#include "tensor.h"
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SharedPtr *malloc_shared_ptr(float *data, int size) {
  SharedPtr *shared_ptr = malloc(sizeof(SharedPtr));
  if (!shared_ptr)
    return NULL;

  shared_ptr->ptr = malloc(size * sizeof(float));
  if (!shared_ptr->ptr) {
    free(shared_ptr);
    return NULL;
  }
  memcpy(shared_ptr->ptr, data, size * sizeof(float));

  shared_ptr->ref_counter = 1;
  return shared_ptr;
}

void free_shared_ptr(SharedPtr **ptr) {
  if (ptr && *ptr) {
    if (--(*ptr)->ref_counter == 0) {
      if ((*ptr)->ptr)
        free((*ptr)->ptr);
      free(*ptr);
      *ptr = NULL;
    }
  }
}

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
    t->grad->ptr[i] = 1.0f;
  }
}

Tensor *malloc_tensor_empty() {
  Tensor *t = malloc(sizeof(Tensor));
  if (!t)
    return NULL;

  memset(t, 0, sizeof(Tensor));
  t->data = NULL;
  t->grad = NULL;
  return t;
}

Tensor *malloc_tensor_shape(const int *shape, int ndim, bool requires_grad) {
  if (ndim < 0 || (ndim > 0 && !shape))
    return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t)
    return NULL;

  memset(t, 0,
         sizeof(Tensor)); // Initialize all members to 0 (NULL for pointers)
  t->ndim = ndim;
  t->requires_grad = requires_grad;

  if (ndim == 0) {
    float zero_val = 0.0f;
    t->data = malloc_shared_ptr(&zero_val, 1);
    if (!t->data) {
      free_tensor(&t);
      return NULL;
    }

    if (requires_grad) {
      float zero_grad_val = 0.0f;
      t->grad = malloc_shared_ptr(&zero_grad_val, 1);
      if (!t->grad) {
        free_tensor(&t);
        return NULL;
      }
    }
    return t;
  }

  t->shape = malloc(ndim * sizeof(int));
  if (!t->shape) {
    free_tensor(&t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  int size = numel(shape, ndim);
  if (size <= 0) {
    free_tensor(&t);
    return NULL;
  }

  t->strides = compute_strides(t->shape, ndim);
  if (!t->strides) {
    free_tensor(&t);
    return NULL;
  }

  float *initial_data = calloc(size, sizeof(float)); // Declare initial_data
  if (!initial_data) {
    free_tensor(&t);
    return NULL;
  }
  t->data = malloc_shared_ptr(initial_data, size);
  free(initial_data); // Free the temporary buffer

  if (requires_grad) {
    float *initial_grad = calloc(size, sizeof(float)); // Declare initial_grad
    if (!initial_grad) {
      free_tensor(&t);
      return NULL;
    }
    t->grad = malloc_shared_ptr(initial_grad, size);
    free(initial_grad);
  }

  return t;
}

Tensor *malloc_tensor_full(const int *shape, int ndim, const int *strides,
                           SharedPtr *data_shared_ptr, bool requires_grad,
                           SharedPtr *grad_shared_ptr) {
  if (ndim < 0 || (ndim > 0 && (!shape || !data_shared_ptr)))
    return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t)
    return NULL;

  memset(t, 0, sizeof(Tensor));
  t->ndim = ndim;
  t->requires_grad = requires_grad;
  t->data = NULL;
  t->grad = NULL;
  t->shape = NULL;
  t->strides = NULL;
  t->grad_fn = NULL;

  int size = 1;
  if (ndim == 0) {
    size = 1;
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

  if (data_shared_ptr) {
    t->data = data_shared_ptr;
  }

  if (requires_grad && grad_shared_ptr) {
    t->grad = grad_shared_ptr;
  }

  return t;
}

void free_tensor(Tensor **t) {
  if (t && *t) {
    if ((*t)->data) {
      free_shared_ptr(&(*t)->data);
    }
    if ((*t)->grad) {
      free_shared_ptr(&(*t)->grad);
    }
    if ((*t)->shape) {
      free((*t)->shape);
    }
    if ((*t)->strides) {
      free((*t)->strides);
    }
    free(*t);
    *t = NULL;
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
    t->data->ptr[i] = 1.0f;
  }
  return t;
}

Tensor *uniform(const int *shape, int ndim, float low, float high,
                bool requires_grad) {
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t)
    return NULL;

  // Seed the random number generator if not already seeded.
  // This is a simple approach; in a real application, seeding should be
  // done once.
  static bool seeded = false;
  if (!seeded) {
    srand(time(NULL));
    seeded = true;
  }

  int size = numel(shape, ndim);
  for (int i = 0; i < size; ++i) {
    t->data->ptr[i] = low + (float)rand() / (RAND_MAX / (high - low));
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

    t->data->ptr[i] = z1;
    if (i + 1 < size) {
      t->data->ptr[i + 1] = z2;
    }
  }
  return t;
}

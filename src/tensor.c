#include "tensor.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SharedPtr *malloc_shared_ptr(float *data, int size) {
  DEBUG_PRINT("[IDRAK_DEBUG] SharedPtr: Allocating new SharedPtr of size %d "
              "bytes for data storage.\n",
              size * sizeof(float));
  SharedPtr *shared_ptr = malloc(sizeof(SharedPtr));
  if (!shared_ptr) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] malloc_shared_ptr: Failed to allocate SharedPtr\n");
    return NULL;
  }

  shared_ptr->ptr = malloc(size * sizeof(float));
  if (!shared_ptr->ptr) {
    DEBUG_PRINT("[IDRAK_DEBUG] malloc_shared_ptr: Failed to allocate data for "
                "SharedPtr\n");
    free(shared_ptr);
    return NULL;
  }
  memcpy(shared_ptr->ptr, data, size * sizeof(float));

  shared_ptr->ref_counter = 1;
  DEBUG_PRINT("[IDRAK_DEBUG] SharedPtr: Successfully allocated at %p, data "
              "pointer at %p. Initial ref_count: %d.\n",
              (void *)shared_ptr, (void *)shared_ptr->ptr,
              shared_ptr->ref_counter);
  return shared_ptr;
}

void free_shared_ptr(SharedPtr **ptr) {
  if (ptr && *ptr) {
    DEBUG_PRINT("[IDRAK_DEBUG] free_shared_ptr: Decrementing ref_counter for "
                "SharedPtr at %p "
                "(current: %d)\n",
                (void *)*ptr, (*ptr)->ref_counter);
    if (--(*ptr)->ref_counter == 0) {
      DEBUG_PRINT("[IDRAK_DEBUG] free_shared_ptr: Ref_counter is 0. Freeing "
                  "data at %p and "
                  "SharedPtr at %p\n",
                  (void *)(*ptr)->ptr, (void *)*ptr);
      if ((*ptr)->ptr)
        free((*ptr)->ptr);
      free(*ptr);
      *ptr = NULL;
    } else {
      DEBUG_PRINT("[IDRAK_DEBUG] free_shared_ptr: SharedPtr at %p still has %d "
                  "references\n",
                  (void *)*ptr, (*ptr)->ref_counter);
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
  DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_empty: Allocating empty Tensor\n");
  Tensor *t = malloc(sizeof(Tensor));
  if (!t) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] malloc_tensor_empty: Failed to allocate Tensor\n");
    return NULL;
  }

  memset(t, 0, sizeof(Tensor));
  t->data = NULL;
  t->grad = NULL;
  DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_empty: Successfully allocated empty "
              "Tensor at %p\n",
              (void *)t);
  return t;
}

Tensor *malloc_tensor_shape(const int *shape, int ndim, bool requires_grad) {
  DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Allocating Tensor with "
              "ndim=%d, requires_grad=%d\n",
              ndim, requires_grad);
  if (ndim < 0 || (ndim > 0 && !shape))
    return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t) {
    DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Failed to allocate Tensor "
                "struct\n");
    return NULL;
  }

  memset(t, 0,
         sizeof(Tensor)); // Initialize all members to 0 (NULL for pointers)
  t->ndim = ndim;
  t->requires_grad = requires_grad;

  if (ndim == 0) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] malloc_tensor_shape: Allocating scalar tensor\n");
    float zero_val = 0.0f;
    t->data = malloc_shared_ptr(&zero_val, 1);
    if (!t->data) {
      DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Failed to allocate data "
                  "for scalar tensor\n");
      free_tensor(&t);
      return NULL;
    }

    if (requires_grad) {
      float zero_grad_val = 0.0f;
      t->grad = malloc_shared_ptr(&zero_grad_val, 1);
      if (!t->grad) {
        DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Failed to allocate "
                    "grad for scalar tensor\n");
        free_tensor(&t);
        return NULL;
      }
    }
    DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Successfully allocated "
                "scalar tensor at %p\n",
                (void *)t);
    return t;
  }

  t->shape = malloc(ndim * sizeof(int));
  if (!t->shape) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] malloc_tensor_shape: Failed to allocate shape array\n");
    free_tensor(&t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  int size = numel(shape, ndim);
  if (size <= 0) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] malloc_tensor_shape: Invalid size (%d) for tensor\n",
        size);
    free_tensor(&t);
    return NULL;
  }

  t->strides = compute_strides(t->shape, ndim);
  if (!t->strides) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] malloc_tensor_shape: Failed to compute strides\n");
    free_tensor(&t);
    return NULL;
  }

  float *initial_data = calloc(size, sizeof(float)); // Declare initial_data
  if (!initial_data) {
    DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Failed to allocate "
                "initial_data buffer\n");
    free_tensor(&t);
    return NULL;
  }
  t->data = malloc_shared_ptr(initial_data, size);
  free(initial_data); // Free the temporary buffer
  if (!t->data) {
    DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Failed to create SharedPtr "
                "for data\n");
    free_tensor(&t);
    return NULL;
  }

  if (requires_grad) {
    float *initial_grad = calloc(size, sizeof(float)); // Declare initial_grad
    if (!initial_grad) {
      DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Failed to allocate "
                  "initial_grad buffer\n");
      free_tensor(&t);
      return NULL;
    }
    t->grad = malloc_shared_ptr(initial_grad, size);
    free(initial_grad);
    if (!t->grad) {
      DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Failed to create "
                  "SharedPtr for grad\n");
      free_tensor(&t);
      return NULL;
    }
  }
  DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_shape: Successfully allocated "
              "Tensor at %p with size %d\n",
              (void *)t, size);
  return t;
}

Tensor *malloc_tensor_full(const int *shape, int ndim, const int *strides,
                           SharedPtr *data_shared_ptr, bool requires_grad,
                           SharedPtr *grad_shared_ptr) {
  DEBUG_PRINT("[IDRAK_DEBUG] Tensor: Allocating new Tensor structure. ndim: "
              "%d, requires_grad: %d.\n",
              ndim, requires_grad);
  if (ndim < 0 || (ndim > 0 && (!shape || !data_shared_ptr)))
    return NULL;

  Tensor *t = malloc(sizeof(Tensor));
  if (!t) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] malloc_tensor_full: Failed to allocate Tensor struct\n");
    return NULL;
  }

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
      DEBUG_PRINT(
          "[IDRAK_DEBUG] malloc_tensor_full: Invalid size (%d) for tensor\n",
          size);
      free(t);
      return NULL;
    }

    t->shape = malloc(ndim * sizeof(int));
    if (!t->shape) {
      DEBUG_PRINT(
          "[IDRAK_DEBUG] malloc_tensor_full: Failed to allocate shape array\n");
      free(t);
      return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    t->strides = malloc(ndim * sizeof(int));
    if (!t->strides) {
      DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_full: Failed to allocate "
                  "strides array\n");
      free(t->shape);
      free(t);
      return NULL;
    }

    if (strides) {
      memcpy(t->strides, strides, ndim * sizeof(int));
    } else {
      int *default_strides = compute_strides(shape, ndim);
      if (!default_strides) {
        DEBUG_PRINT("[IDRAK_DEBUG] malloc_tensor_full: Failed to compute "
                    "default strides\n");
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
    DEBUG_PRINT(
        "[IDRAK_DEBUG] Tensor: Referencing existing data SharedPtr at %p.\n",
        (void *)data_shared_ptr);
  }

  if (requires_grad && grad_shared_ptr) {
    t->grad = grad_shared_ptr;
    DEBUG_PRINT(
        "[IDRAK_DEBUG] Tensor: Referencing existing grad SharedPtr at %p.\n",
        (void *)grad_shared_ptr);
  }
  DEBUG_PRINT(
      "[IDRAK_DEBUG] Tensor: Successfully allocated Tensor structure at %p.\n",
      (void *)t);
  return t;
}

void free_tensor(Tensor **t) {
  if (t && *t) {
    DEBUG_PRINT("[IDRAK_DEBUG] free_tensor: Freeing Tensor at %p\n",
                (void *)*t);
    if ((*t)->data) {
      DEBUG_PRINT("[IDRAK_DEBUG] free_tensor: Freeing data SharedPtr for "
                  "Tensor at %p\n",
                  (void *)*t);
      free_shared_ptr(&(*t)->data);
    }
    if ((*t)->grad) {
      DEBUG_PRINT("[IDRAK_DEBUG] free_tensor: Freeing grad SharedPtr for "
                  "Tensor at %p\n",
                  (void *)*t);
      free_shared_ptr(&(*t)->grad);
    }
    if ((*t)->shape) {
      DEBUG_PRINT(
          "[IDRAK_DEBUG] free_tensor: Freeing shape array for Tensor at %p\n",
          (void *)*t);
      free((*t)->shape);
    }
    if ((*t)->strides) {
      DEBUG_PRINT(
          "[IDRAK_DEBUG] free_tensor: Freeing strides array for Tensor at %p\n",
          (void *)*t);
      free((*t)->strides);
    }
    DEBUG_PRINT("[IDRAK_DEBUG] free_tensor: Tensor at %p successfully freed\n",
                (void *)*t);
    free(*t);
    *t = NULL;
  }
}

Tensor *zeros(const int *shape, int ndim, bool requires_grad) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] zeros: Creating zero tensor (ndim=%d, requires_grad=%d)\n",
      ndim, requires_grad);
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t) {
    DEBUG_PRINT("[IDRAK_DEBUG] zeros: Failed to create zero tensor\n");
    return NULL;
  }

  // malloc_tensor_shape already sets data to 0, so nothing more to do here.
  DEBUG_PRINT("[IDRAK_DEBUG] zeros: Successfully created zero tensor at %p\n",
              (void *)t);
  return t;
}

Tensor *ones(const int *shape, int ndim, bool requires_grad) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] ones: Creating ones tensor (ndim=%d, requires_grad=%d)\n",
      ndim, requires_grad);
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t) {
    DEBUG_PRINT("[IDRAK_DEBUG] ones: Failed to create ones tensor\n");
    return NULL;
  }

  int size = numel(shape, ndim);
  for (int i = 0; i < size; ++i) {
    t->data->ptr[i] = 1.0f;
  }
  DEBUG_PRINT("[IDRAK_DEBUG] ones: Successfully created ones tensor at %p\n",
              (void *)t);
  return t;
}

Tensor *uniform(const int *shape, int ndim, float low, float high,
                bool requires_grad) {
  DEBUG_PRINT("[IDRAK_DEBUG] uniform: Creating uniform tensor (ndim=%d, "
              "low=%.2f, high=%.2f, requires_grad=%d)\n",
              ndim, low, high, requires_grad);
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t) {
    DEBUG_PRINT("[IDRAK_DEBUG] uniform: Failed to create uniform tensor\n");
    return NULL;
  }

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
  DEBUG_PRINT(
      "[IDRAK_DEBUG] uniform: Successfully created uniform tensor at %p\n",
      (void *)t);
  return t;
}

Tensor *randn(const int *shape, int ndim, int seed, bool requires_grad) {
  DEBUG_PRINT("[IDRAK_DEBUG] randn: Creating random normal tensor (ndim=%d, "
              "seed=%d, requires_grad=%d)\n",
              ndim, seed, requires_grad);
  Tensor *t = malloc_tensor_shape(shape, ndim, requires_grad);
  if (!t) {
    DEBUG_PRINT("[IDRAK_DEBUG] randn: Failed to create random normal tensor\n");
    return NULL;
  }

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
  DEBUG_PRINT(
      "[IDRAK_DEBUG] randn: Successfully created random normal tensor at %p\n",
      (void *)t);
  return t;
}

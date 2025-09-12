#include "tensor.h"
#include "logger.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool is_contiguous(Tensor* t) {
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

int numel(const int* shape, int ndim) {
    if (ndim == 0)
        return 1;
    if (ndim < 0 || !shape)
        return 0;
    int size = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0)
            return 0;
        size *= shape[i];
    }
    return size;
}

int* compute_strides(const int* shape, int ndim) {
    if (ndim <= 0 || !shape)
        return NULL;

    int* strides = malloc(ndim * sizeof(int));
    if (!strides)
        return NULL;

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

void set_ones_grad(Tensor* t) {
    int size = numel(t->shape, t->ndim);
    for (int i = 0; i < size; ++i) {
        ((float*)t->grad->elems)[i] = 1.0f;
    }
}

Tensor* tmalloc_shape(const int* shape, int ndim, Dtype dtype, Device device, bool requires_grad) {
    LOG_INFO("ALLOC: tmalloc_shape: Allocating Tensor with ndim=%d, requires_grad=%d", ndim, requires_grad);
    if (ndim < 0 || (ndim > 0 && !shape))
        return NULL;

    Tensor* t = malloc(sizeof(Tensor));
    if (!t) {
        LOG_ERROR("tmalloc_shape: Failed to allocate Tensor struct.");
        return NULL;
    }

    t->ndim = ndim;
    t->requires_grad = requires_grad;

    if (ndim == 0) {
        LOG_INFO("ALLOC: tmalloc_shape: Allocating scalar tensor");
        t->data = palloc(NULL, 1, dtype, device);
        if (t->data->elems == NULL) {
            LOG_ERROR("tmalloc_shape: Failed to allocate data for scalar tensor.");
            free(t);
            return NULL;
        }

        if (requires_grad) {
            t->grad = palloc(NULL, 1, dtype, device);
            if (t->grad->elems == NULL) {
                LOG_ERROR("tmalloc_shape: Failed to allocate grad for scalar tensor.");
                if (t->data != NULL) {
                    pfree(t->data);
                }
                free(t);
                return NULL;
            }
        }
        LOG_INFO("ALLOC: tmalloc_shape: Successfully allocated scalar tensor at %p", (void*)t);
        return t;
    }

    t->shape = malloc(ndim * sizeof(int));
    if (!t->shape) {
        LOG_ERROR("tmalloc_shape: Failed to allocate shape array.");
        tfree(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));

    int size = numel(shape, ndim);
    if (size <= 0) {
        LOG_ERROR("tmalloc_shape: Invalid size (%d) for tensor.", size);
        tfree(t);
        return NULL;
    }

    t->strides = compute_strides(t->shape, ndim);
    if (!t->strides) {
        LOG_ERROR("tmalloc_shape: Failed to compute strides.");
        if (t->shape) {
            free(t->shape);
        }
        free(t);
        return NULL;
    }

    t->data = palloc(NULL, size, dtype, device);
    LOG_INFO("tmalloc_shape: palloc returned. t->data->elems = %p", t->data->elems);
    if (t->data->elems == NULL) {
        LOG_ERROR("tmalloc_shape: Failed to allocate SharedPtr for data.");
        if (t->shape)
            free(t->shape);
        if (t->strides)
            free(t->strides);
        free(t);
        return NULL;
    }

    for (int i = 0; i < size; ++i) {
        ((float*)t->data->elems)[i] = 0.0f;
    }

    if (requires_grad) {
        t->grad = palloc(NULL, size, dtype, device);
        if (t->grad->elems == NULL) {
            LOG_ERROR("tmalloc_shape: Failed to allocate SharedPtr for grad.");
            if (t->data != NULL) {
                pfree(t->data);
            }
            if (t->shape) {
                free(t->shape);
            }
            if (t->strides) {
                free(t->strides);
            }
            free(t);
            return NULL;
        }
        for (int i = 0; i < size; ++i) {
            ((float*)t->grad->elems)[i] = 0.0f;
        }
    }
    LOG_INFO("ALLOC: tmalloc_shape: Successfully allocated Tensor at %p with size %d", (void*)t, size);
    return t;
}

Tensor* tmalloc_full(const int* shape, int ndim, const int* strides, Dtype dtype, Device device, shared_ptr* data_shared_ptr, bool requires_grad, shared_ptr* grad_shared_ptr) {
    LOG_INFO("ALLOC: tmalloc_full: Allocating new Tensor structure. ndim: %d, requires_grad: %d.", ndim, requires_grad);
    if (ndim < 0 || (ndim > 0 && (!shape || data_shared_ptr->elems == NULL)))
        return NULL;

    Tensor* t = malloc(sizeof(Tensor));
    if (!t) {
        LOG_ERROR("tmalloc_full: Failed to allocate Tensor struct.");
        return NULL;
    }

    t->ndim = ndim;
    t->requires_grad = requires_grad;

    int size = 1;
    if (ndim == 0) {
        size = 1;
    } else {
        size = numel(shape, ndim);
        if (size <= 0) {
            LOG_ERROR("tmalloc_full: Invalid size (%d) for tensor.", size);
            free(t);
            return NULL;
        }

        t->shape = malloc(ndim * sizeof(int));
        if (!t->shape) {
            LOG_ERROR("tmalloc_full: Failed to allocate shape array.");
            free(t);
            return NULL;
        }
        memcpy(t->shape, shape, ndim * sizeof(int));

        t->strides = malloc(ndim * sizeof(int));
        if (!t->strides) {
            LOG_ERROR("tmalloc_full: Failed to allocate strides array.");
            free(t->shape);
            free(t);
            return NULL;
        }

        if (strides) {
            memcpy(t->strides, strides, ndim * sizeof(int));
        } else {
            int* default_strides = compute_strides(shape, ndim);
            if (!default_strides) {
                LOG_ERROR("tmalloc_full: Failed to compute default strides.");
                free(t->strides);
                free(t->shape);
                free(t);
                return NULL;
            }
            memcpy(t->strides, default_strides, ndim * sizeof(int));
            free(default_strides);
        }
    }

    if (data_shared_ptr != NULL && data_shared_ptr->elems != NULL) {
        t->data = data_shared_ptr;
        t->data->ref_counter++;
        LOG_INFO("ALLOC: Tensor: Referencing existing data SharedPtr at %p. "
                 "Incremented ref_counter to %d.",
                 (void*)t->data->elems, t->data->ref_counter);
    }

    if (requires_grad && grad_shared_ptr != NULL && grad_shared_ptr->elems != NULL) {
        t->grad = grad_shared_ptr;
        t->grad->ref_counter++;
        LOG_INFO("ALLOC: Tensor: Referencing existing grad SharedPtr at %p. "
                 "Incremented ref_counter to %d.",
                 (void*)t->grad->elems, t->grad->ref_counter);
    }
    LOG_INFO("ALLOC: Tensor: Successfully allocated Tensor structure at %p.", (void*)t);
    return t;
}

void tfree(Tensor* t) {
    if (t) {
        LOG_INFO("FREE: free_tensor: Freeing Tensor at %p", (void*)t);
        if (t->data != NULL) {
            LOG_INFO("FREE: free_tensor: Freeing data SharedPtr for Tensor at %p", (void*)t->data->elems);
            pfree(t->data);
        }
        if (t->grad != NULL) {
            LOG_INFO("FREE: free_tensor: Freeing grad SharedPtr for Tensor at %p", (void*)t->grad->elems);
            pfree(t->grad);
        }
        if (t->shape) {
            LOG_INFO("FREE: free_tensor: Freeing shape array for Tensor at %p", (void*)t);
            free(t->shape);
        }
        if (t->strides) {
            LOG_INFO("FREE: free_tensor: Freeing strides array for Tensor at %p", (void*)t);
            free(t->strides);
        }
        LOG_INFO("FREE: free_tensor: Tensor at %p successfully freed", (void*)t);
        free(t);
    }
}

Tensor* zeros(const int* shape, int ndim, Dtype dtype, Device device, bool requires_grad) {
    LOG_INFO("DEBUG: zeros: Creating zero tensor (ndim=%d, requires_grad=%d)", ndim, requires_grad);
    Tensor* t = tmalloc_shape(shape, ndim, dtype, device, requires_grad);
    if (!t) {
        LOG_ERROR("zeros: Failed to create zero tensor.");
        return NULL;
    }

    // tmalloc_shape already sets data to 0, so nothing more to do here.
    LOG_INFO("DEBUG: zeros: Successfully created zero tensor at %p", (void*)t);
    return t;
}

Tensor* ones(const int* shape, int ndim, Dtype dtype, Device device, bool requires_grad) {
    LOG_INFO("DEBUG: ones: Creating ones tensor (ndim=%d, requires_grad=%d)", ndim, requires_grad);
    Tensor* t = tmalloc_shape(shape, ndim, dtype, device, requires_grad);
    if (!t) {
        LOG_ERROR("ones: Failed to create ones tensor.");
        return NULL;
    }

    int size = numel(shape, ndim);
    for (int i = 0; i < size; ++i) {
        ((float*)t->data->elems)[i] = 1.0f;
    }
    LOG_INFO("DEBUG: ones: Successfully created ones tensor at %p", (void*)t);
    return t;
}

Tensor* uniform(const int* shape, int ndim, Dtype dtype, Device device, float low, float high, bool requires_grad) {
    LOG_INFO("DEBUG: uniform: Creating uniform tensor (ndim=%d, low=%.2f, high=%.2f, requires_grad=%d)", ndim, low, high, requires_grad);
    Tensor* t = tmalloc_shape(shape, ndim, dtype, device, requires_grad);
    if (!t) {
        LOG_ERROR("uniform: Failed to create uniform tensor.");
        return NULL;
    }

    static bool seeded = false;
    if (!seeded) {
        srand(time(NULL));
        seeded = true;
    }

    int size = numel(shape, ndim);
    for (int i = 0; i < size; ++i) {
        ((float*)t->data->elems)[i] = low + (float)rand() / (RAND_MAX / (high - low));
    }
    LOG_INFO("DEBUG: uniform: Successfully created uniform tensor at %p", (void*)t);
    return t;
}

Tensor* randn(const int* shape, int ndim, Dtype dtype, Device device, int seed, bool requires_grad) {
    LOG_INFO("DEBUG: randn: Creating random normal tensor (ndim=%d, seed=%d, requires_grad=%d)", ndim, seed, requires_grad);
    Tensor* t = tmalloc_shape(shape, ndim, dtype, device, requires_grad);
    if (!t) {
        LOG_ERROR("randn: Failed to create random normal tensor.");
        return NULL;
    }

    srand(seed); // Seed for reproducibility

    int size = numel(shape, ndim);
    for (int i = 0; i < size; i += 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;

        float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * M_PI * u2);

        ((float*)t->data->elems)[i] = z1;
        if (i + 1 < size) {
            ((float*)t->data->elems)[i + 1] = z2;
        }
    }
    LOG_INFO("DEBUG: randn: Successfully created random normal tensor at %p", (void*)t);
    return t;
}

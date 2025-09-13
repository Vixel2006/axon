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
            if (t->strides[i] != expected_stride) {
                LOG_INFO("Tensor not contiguous at dim %d (stride=%d, expected=%d)", i, t->strides[i], expected_stride);
                return false;
            }
            expected_stride *= t->shape[i];
        }
    }
    LOG_INFO("Tensor is contiguous");
    return true;
}

int numel(const int* shape, int ndim) {
    if (ndim == 0)
        return 1;
    if (ndim < 0 || !shape)
        return 0;

    int size = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) {
            LOG_WARN("Invalid shape[%d] = %d, returning 0", i, shape[i]);
            return 0;
        }
        size *= shape[i];
    }
    LOG_INFO("numel computed: %d", size);
    return size;
}

int* compute_strides(const int* shape, int ndim) {
    if (ndim <= 0 || !shape)
        return NULL;

    int* strides = malloc(ndim * sizeof(int));
    if (!strides) {
        LOG_ERROR("Failed to allocate strides for ndim=%d", ndim);
        return NULL;
    }

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    LOG_INFO("Strides allocated at %p for ndim=%d", strides, ndim);
    return strides;
}

Storage* smalloc(float* data, int size) {
    Storage* s = malloc(sizeof(Storage));
    if (!s) {
        LOG_ERROR("Failed to allocate Storage");
        return NULL;
    }
    s->size = size;
    s->data = malloc(sizeof(float) * size);
    if (!s->data) {
        LOG_ERROR("Failed to allocate Storage data");
        free(s);
        return NULL;
    }

    s->data = data;

    s->counter = 1;

    LOG_INFO("Storage allocated at %p with size=%d", s, size);
    return s;
}

void sfree(Storage* s) {
    if (!s)
        return;
    s->counter--;
    LOG_INFO("sfree called, counter=%d for Storage %p", s->counter, s);
    if (s->counter <= 0) {
        if (s->data) {
            free(s->data);
            LOG_INFO("Storage data freed at %p", s->data);
        }
        free(s);
    }
}

Tensor* tmalloc(int* shape, int ndim, Device device, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));
    if (!t) {
        LOG_ERROR("Failed to allocate Tensor");
        return NULL;
    }

    t->ndim = ndim;
    t->shape = malloc(sizeof(int) * ndim);
    if (!t->shape) {
        LOG_ERROR("Failed to allocate Tensor shape");
        free(t);
        return NULL;
    }

    for (int i = 0; i < t->ndim; ++i)
        t->shape[i] = shape[i];

    t->strides = compute_strides(t->shape, t->ndim);
    t->device = device;
    t->requires_grad = requires_grad;
    t->data = NULL; // initialize pointer to NULL

    LOG_INFO("Tensor allocated at %p (ndim=%d, device=%d, requires_grad=%d)", t, t->ndim, t->device, t->requires_grad);
    return t;
}

void tfree(Tensor* t) {
    if (!t)
        return;

    LOG_INFO("Freeing Tensor at %p", t);

    if (t->shape) {
        free(t->shape);
        LOG_INFO("Freed shape at %p", t->shape);
    }

    if (t->strides) {
        free(t->strides);
        LOG_INFO("Freed strides at %p", t->strides);
    }

    if (t->data) {
        sfree(t->data);
    }

    free(t);
}

#include "tensor.h"
#include "logger.h"
#include <cuda_runtime.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool is_contiguous(Tensor* t)
{
    int expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; --i)
    {
        if (t->shape[i] > 1)
        {
            if (t->strides[i] != expected_stride)
            {
                LOG_INFO("Tensor not contiguous at dim %d (stride=%d, expected=%d)", i,
                         t->strides[i], expected_stride);
                return false;
            }
            expected_stride *= t->shape[i];
        }
    }
    LOG_INFO("Tensor is contiguous");
    return true;
}

bool shapes_equal(const int* shape1, int ndim1, const int* shape2, int ndim2)
{
    if (ndim1 != ndim2) return false;
    for (int i = 0; i < ndim1; ++i)
    {
        if (shape1[i] != shape2[i]) return false;
    }
    return true;
}

int numel(const int* shape, int ndim)
{
    if (ndim == 0) return 1;
    if (ndim < 0 || !shape) return 0;

    int size = 1;
    for (int i = 0; i < ndim; ++i)
    {
        if (shape[i] <= 0)
        {
            LOG_WARN("Invalid shape[%d] = %d, returning 0", i, shape[i]);
            return 0;
        }
        size *= shape[i];
    }
    LOG_INFO("numel computed: %d", size);
    return size;
}

int* compute_strides(const int* shape, int ndim)
{
    if (ndim <= 0 || !shape) return NULL;

    int* strides = malloc(ndim * sizeof(int));
    if (!strides)
    {
        LOG_ERROR("Failed to allocate strides for ndim=%d", ndim);
        return NULL;
    }

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    LOG_INFO("Strides allocated at %p for ndim=%d", strides, ndim);
    return strides;
}

Storage* smalloc(float* data, int size, Device device)
{
    Storage* s = malloc(sizeof(Storage));

    if (!s)
    {
        LOG_ERROR("Failed to allocate Storage");
        return NULL;
    }

    s->size = size;

    if (device == CPU)
    {
        s->data = malloc(sizeof(float) * size);

        if (!s->data)
        {
            LOG_ERROR("Failed to allocate Storage data on cpu.");
            free(s);
            return NULL;
        }

        if (data != NULL)
        {
            for (int i = 0; i < size; ++i)
                s->data[i] = data[i];
        }
        else
        {
            for (int i = 0; i < size; ++i)
                s->data[i] = 0.0f;
        }
    }
    else
    {
        cudaError_t err = cudaMalloc((void**) &s->data, size);

        if (err != cudaSuccess)
        {
            LOG_ERROR("Failed to allocate Storage data on cuda device.");
            free(s);
            return NULL;
        }

        cudaMemcpy(s->data, data, size, cudaMemcpyHostToDevice);
    }

    s->counter = 1;

    LOG_INFO("Storage allocated at %p with size=%d", s, size);
    return s;
}

void sfree(Storage* s, Device device)
{
    if (!s) return;
    s->counter--;
    LOG_INFO("sfree called, counter=%d for Storage %p", s->counter, s);
    if (s->counter <= 0)
    {
        if (s->data)
        {
            LOG_INFO("Storage data freed at %p", s->data);
            if (device == CPU)
            {
                SAFE_FREE(&s->data, free);
            }
            else
            {
                cudaFree(s->data);
                s->data = NULL;
            }
        }
        SAFE_FREE(&s, free);
    }
}

void gmalloc(Tensor* t, float init)
{
    int size = numel(t->shape, t->ndim);

    if (t->grad)
    {
        sfree(t->grad, t->device);
        t->grad = NULL;
    }

    t->grad = malloc(sizeof(Storage));
    if (!t->grad)
    {
        LOG_ERROR("Failed to allocate Storage for grad");
        return;
    }

    t->grad->size = size;

    if (t->device == CPU)
    {
        t->grad->data = malloc(sizeof(float) * size);
        if (!t->grad->data)
        {
            LOG_ERROR("Failed to allocate Storage data for grad on cpu.");
            sfree(t->grad, t->device);
            t->grad = NULL;
            return;
        }

        for (int i = 0; i < size; ++i)
        {
            t->grad->data[i] = init;
        }
    }
    else
    {
        cudaError_t err = cudaMalloc((void**) &t->grad->data, size);
        if (err != cudaSuccess)
        {
            LOG_ERROR("Failed to allocate Storage data for grad on cuda device.");
            sfree(t->grad, t->device);
            t->grad = NULL;
            return;
        }
        cudaMemset(t->grad->data, init, size);
    }

    t->grad->counter = 1;
    LOG_INFO("Storage allocated for grad at %p with size=%d", t->grad, size);
}

void gfree(Tensor* t)
{
    if (t->grad)
    {
        sfree(t->grad, t->device);
        t->grad = NULL;
    }
}

Tensor* tmalloc(int* shape, int ndim, Device device, bool requires_grad)
{
    Tensor* t = malloc(sizeof(Tensor));
    if (!t)
    {
        LOG_ERROR("Failed to allocate Tensor");
        return NULL;
    }

    t->ndim = ndim;
    t->shape = malloc(sizeof(int) * ndim);
    if (!t->shape)
    {
        LOG_ERROR("Failed to allocate Tensor shape");
        SAFE_FREE(&t, free);
        return NULL;
    }

    for (int i = 0; i < t->ndim; ++i)
        t->shape[i] = shape[i];

    t->strides = compute_strides(t->shape, t->ndim);
    LOG_INFO("tmalloc: t=%p, t->shape=%p, t->strides=%p", t, t->shape, t->strides);
    t->device = device;
    t->requires_grad = requires_grad;
    t->data = NULL;
    t->grad = NULL;

    LOG_INFO("Tensor allocated at %p (ndim=%d, device=%d, requires_grad=%d)", t, t->ndim, t->device,
             t->requires_grad);
    return t;
}

void tfree(Tensor* t)
{
    if (!t) return;

    LOG_INFO("tfree: t=%p, t->shape=%p, t->strides=%p", t, t->shape, t->strides);

    LOG_INFO("Freeing Tensor at %p", t);

    if (t->shape)
    {
        LOG_INFO("Freed shape at %p", t->shape);
        SAFE_FREE(&t->shape, free);
    }

    if (t->strides)
    {
        LOG_INFO("Freed strides at %p", t->strides);
        SAFE_FREE(&t->strides, free);
    }

    if (t->data)
    {
        sfree(t->data, t->device);
        t->data = NULL;
    }

    if (t->grad)
    {
        sfree(t->grad, t->device);
        t->grad = NULL;
    }

    SAFE_FREE(&t, free);
}

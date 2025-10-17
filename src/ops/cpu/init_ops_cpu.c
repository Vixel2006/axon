#include "logger.h"
#include "ops/init_ops.h"
#include <stdio.h>
#include <string.h>

#define PREP(shape, ndim, data_ptr, size_var)                                                      \
    do                                                                                             \
    {                                                                                              \
        size_var = numel(shape, ndim);                                                             \
        data_ptr = malloc(sizeof(float) * size_var);                                               \
        if (!data_ptr)                                                                             \
        {                                                                                          \
            fprintf(stderr, "malloc failed\n");                                                    \
        }                                                                                          \
    } while (0)

void zeros(Tensor* t)
{
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);

    for (int i = 0; i < size; ++i)
    {
        data[i] = 0.0f;
    }

    // t->data = malloc(sizeof(Storage));

    t->data = smalloc(data, size, t->device);

    SAFE_FREE(&data, free);

    gmalloc(t, 0.0f);

    if (!t->data)
    {
        SAFE_FREE(&t, tfree);
        return;
    }
    LOG_INFO("OP: zeros: Initialized tensor t: data=%p, grad=%p", (void*) t->data->data,
             (void*) t->grad->data->data);
}

void ones(Tensor* t)
{
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);

    for (int i = 0; i < size; ++i)
    {
        data[i] = 1.0;
    }

    t->data = smalloc(data, size, t->device);
    SAFE_FREE(&data, free);

    gmalloc(t, 0.0f);

    if (!t->data)
    {
        SAFE_FREE(&t, tfree);
        return;
    }
}

void randn(Tensor* t)
{
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);
    srand((unsigned int) time(NULL));

    for (int i = 0; i < size; ++i)
    {
        data[i] = (float) rand() / RAND_MAX;
    }

    t->data = smalloc(data, size, t->device);
    SAFE_FREE(&data, free);

    gmalloc(t, 0.0f);

    if (!t->data)
    {
        SAFE_FREE(&t, tfree);
        return;
    }
}

void uniform(Tensor* t, float low, float high)
{
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);
    srand((unsigned int) time(NULL));

    for (int i = 0; i < size; ++i)
    {
        data[i] = low + (high - low) * ((float) rand() / RAND_MAX);
    }

    t->data = smalloc(data, size, t->device);
    SAFE_FREE(&data, free);

    gmalloc(t, 0.0f);

    if (!t->data)
    {
        SAFE_FREE(&t, tfree);
        return;
    }
}

void from_data(Tensor* t, float* data)
{
    int size = numel(t->shape, t->ndim);

    if (t->data)
    {
        sfree(t->data, t->device);
        t->data = NULL;
    }

    t->data = smalloc(data, size, t->device);

    if (t->requires_grad)
    {
        gmalloc(t, 0.0f);
    }

    if (!t->data)
    {
        SAFE_FREE(&t, tfree);
        return;
    }
}

void borrow(Tensor* t, Storage* data, Tensor* grad)
{
    t->data = data;
    if (data)
    {
        data->counter += 1;
    }

    if (grad)
    {
        if (!t->grad)
        {
            t->grad = tmalloc(t->shape, t->ndim, t->device, false);
            if (!t->grad)
            {
                LOG_ERROR("Failed to allocate grad tensor in borrow");
                return;
            }
        }

        for (int i = 0; i < t->ndim; i++)
        {
            t->grad->strides[i] = t->strides[i];
        }

        if (t->grad->data)
        {
            sfree(t->grad->data, t->device);
        }
        t->grad->data = grad->data;
        if (grad->data)
        {
            grad->data->counter++;
        }
    }
}

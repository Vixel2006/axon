#include "ops/init_ops.h"
#include <stdio.h>

#define PREP(shape, ndim, data_ptr, size_var)                                                                                                                                                                                                                                                              \
    do {                                                                                                                                                                                                                                                                                                   \
        size_var = numel(shape, ndim);                                                                                                                                                                                                                                                                     \
        data_ptr = malloc(sizeof(float) * size_var);                                                                                                                                                                                                                                                       \
        if (!data_ptr) {                                                                                                                                                                                                                                                                                   \
            fprintf(stderr, "malloc failed\n");                                                                                                                                                                                                                                                            \
        }                                                                                                                                                                                                                                                                                                  \
    } while (0)

void zeros(Tensor* t) {
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);

    for (int i = 0; i < size; ++i) {
        data[i] = 0.0;
    }

    t->data = smalloc(data, size);

    if (!t->data) {
        SAFE_FREE(t, tfree);
        return;
    }
}

void ones(Tensor* t) {
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);

    for (int i = 0; i < size; ++i) {
        data[i] = 1.0;
    }

    t->data = smalloc(data, size);

    if (!t->data) {
        SAFE_FREE(t, tfree);
        return;
    }
}

void randn(Tensor* t) {
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);
    srand((unsigned int)time(NULL));

    for (int i = 0; i < size; ++i) {
        data[i] = (float)rand() / RAND_MAX;
    }

    t->data = smalloc(data, size);

    if (!t->data) {
        SAFE_FREE(t, tfree);
        return;
    }
}

void uniform(Tensor* t, float low, float high) {
    int size;
    float* data;
    PREP(t->shape, t->ndim, data, size);
    srand((unsigned int)time(NULL));

    for (int i = 0; i < size; ++i) {
        data[i] = low + (high - low) * ((float)rand() / RAND_MAX);
    }

    t->data = smalloc(data, size);

    if (!t->data) {
        SAFE_FREE(t, tfree);
        return;
    }
}

void from_data(Tensor* t, float* data) {
    int size = numel(t->shape, t->ndim);

    t->data = smalloc(data, size);

    if (!t->data) {
        SAFE_FREE(t, tfree);
        return;
    }
}

void borrow(Tensor* t, Storage* data) {
    t->data = data;
    data->counter += 1;
}

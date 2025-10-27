#include "ops/cpu/init.h"

void uniform(Tensor* t, float low, float high)
{
    LOG_INFO("uniform: Entering function with low=%.2f, high=%.2f", low, high);
    int size = numel(t->shape, t->ndim);
    float* data = malloc(sizeof(float) * size);
    if (!data)
    {
        fprintf(stderr, "malloc failed\n");
        return;
    }
    srand((unsigned int) time(NULL));

    for (int i = 0; i < size; ++i)
    {
        data[i] = low + (high - low) * ((float) rand() / RAND_MAX);
    }

    t->data = smalloc(data, size, t->device);
    SAFE_FREE(&data, free);

    if (!t->data)
    {
        return;
    }

    if (t->requires_grad)
    {
        gmalloc(t, 0.0f);
    }
}

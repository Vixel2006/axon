#include "ops/cpu/init.h"

void ones(Tensor* t)
{
    LOG_INFO("ones: Entering function");
    int size = numel(t->shape, t->ndim);
    float* data = malloc(sizeof(float) * size);
    if (!data)
    {
        fprintf(stderr, "malloc failed\n");
        return;
    }

    for (int i = 0; i < size; ++i)
    {
        data[i] = 1.0;
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

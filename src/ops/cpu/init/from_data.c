#include "ops/cpu/init.h"

void from_data(Tensor* t, float* data)
{
    LOG_INFO("from_data: Entering function");
    int size = numel(t->shape, t->ndim);

    if (t->data)
    {
        sfree(t->data, t->device);
        t->data = NULL;
    }

    t->data = smalloc(data, size, t->device);

    if (!t->data)
    {
        return;
    }

    if (t->requires_grad)
    {
        gmalloc(t, 0.0f);
    }
}

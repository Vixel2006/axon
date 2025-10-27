#include "ops/cpu/init.h"

void zeros(Tensor* t)
{
    LOG_INFO("zeros: Entering function");
    int size = numel(t->shape, t->ndim);
    t->data = smalloc(NULL, size, t->device);

    if (!t->data)
    {
        return;
    }

    if (t->requires_grad)
    {
        gmalloc(t, 0.0f);
    }

    LOG_INFO("zeros: Initialized tensor t: data=%p, grad=%p", (void*) t->data->data,
             (t->grad && t->grad->data) ? (void*) t->grad->data->data : NULL);
}

#include "ops/cpu/init.h"

void borrow(Tensor* t, Storage* data, Tensor* grad)
{
    LOG_INFO("borrow: Entering function");
    t->data = data;
    t->data->counter++;

    if (grad)
    {
        if (!t->grad)
        {
            t->grad = tmalloc(grad->shape, grad->ndim, grad->device, false);
            if (!t->grad)
            {
                LOG_ERROR("Failed to allocate grad tensor in borrow");
                return;
            }
        }

        if (t->grad->data)
        {
            sfree(t->grad->data, t->device);
        }
        t->grad->data = grad->data;
        t->grad->data->counter++;
    }
}

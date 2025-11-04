#include "ops/cuda/init.h"
#include "logger.h"
#include "tensor.h"

void from_data_cuda(Tensor* t, float* data)
{
    LOG_INFO("from_data_cuda: Entering function");
    int size = numel(t->shape, t->ndim);

    if (t->data)
    {
        sfree(t->data, t->device);
        t->data = NULL;
    }

    // smalloc will handle copying data from host to device if data is not NULL
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

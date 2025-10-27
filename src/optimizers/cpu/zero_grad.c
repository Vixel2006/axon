#include "logger.h"
#include "optimizers/optimizers.h"
#include "utils.h"
#include <string.h>

void zero_grad_cpu(Tensor** parameters, int num_parameters)
{
    LOG_INFO("zero_grad_cpu: Entering function with num_parameters=%d", num_parameters);

    for (int i = 0; i < num_parameters; ++i)
    {
        Tensor* t = parameters[i];

        if (t == NULL)
        {
            LOG_WARN("zero_grad: Parameter at index %d is NULL, skipping zero_grad.", i);
            continue;
        }

        if (!t->requires_grad)
        {
            continue;
        }

        if (t->grad == NULL)
        {
            LOG_WARN("zero_grad: Parameter at index %d requires gradients but "
                     "t->grad (SharedData) is NULL. Cannot zero gradient.",
                     i);
            continue;
        }

        size_t size = numel(t->shape, t->ndim);
        if (size > 0 && t->grad && t->grad->data && t->grad->data->data)
        {
            // Set all elements in the gradient data to 0.0f
            for (size_t j = 0; j < size; ++j)
            {
                t->grad->data->data[j] = 0.0f;
            }
            LOG_INFO("zero_grad: Zeroed gradient for parameter %d (size=%zu)", i, size);
        }
    }
}

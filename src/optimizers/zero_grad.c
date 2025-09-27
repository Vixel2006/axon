#include "logger.h"
#include "optimizers/optimizers.h"
#include "utils.h"
#include <string.h>

void zero_grad(Tensor** parameters, int num_parameters) {
    LOG_INFO("zero_grad: Zeroing gradients for %d parameters", num_parameters);

    for (int i = 0; i < num_parameters; ++i) {
        Tensor* t = parameters[i];

        if (t == NULL) {
            LOG_WARN("zero_grad: Parameter at index %d is NULL, skipping zero_grad.", i);
            continue;
        }

        if (!t->requires_grad) {
            continue;
        }

        if (t->grad->data == NULL) {
            LOG_WARN("zero_grad: Parameter at index %d requires gradients but "
                     "t->grad (SharedData) is NULL. Cannot zero gradient.",
                     i);
            continue;
        }

        if (t->grad->data == NULL) {
            LOG_WARN("zero_grad: Parameter at index %d requires gradients but "
                     "t->grad->elems is NULL. Cannot zero gradient.",
                     i);
            continue;
        }

        size_t size = numel(t->shape, t->ndim);
        if (size > 0 && t->grad && t->grad->data) { // Ensure grad and its data exist
            // Set all elements in the gradient data to 0.0f
            for (size_t j = 0; j < size; ++j) {
                t->grad->data[j] = 0.0f;
            }
            LOG_INFO("zero_grad: Zeroed gradient for parameter %d (size=%zu)", i, size);
        }
    }
}

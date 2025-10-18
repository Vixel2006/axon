#include "logger.h"
#include "optimizers/optimizers.h"
#include "utils.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void sgd_cpu(Tensor** params, int num_params, float lr)
{
    __m256 lr_vec = _mm256_set1_ps(lr);

    for (int i = 0; i < num_params; ++i)
    {
        if (!params[i] || !params[i]->requires_grad || !params[i]->grad || !params[i]->grad->data ||
            !params[i]->grad->data->data || !params[i]->data->data)
        {
            LOG_WARN(
                "sgd: Skipping parameter %d due to invalid tensor, missing grad, or missing data.",
                i);
            continue;
        }

        int num_elements = numel(params[i]->shape, params[i]->ndim);

        // --- Contiguous Path ---
        if (is_contiguous(params[i]))
        {
            float* data_ptr = params[i]->data->data;
            float* grad_ptr = params[i]->grad->data->data;

            int j = 0;
            for (; j + SIMD_WIDTH - 1 < num_elements; j += SIMD_WIDTH)
            {
                __m256 data_vec = _mm256_loadu_ps(data_ptr + j);
                __m256 grad_vec = _mm256_loadu_ps(grad_ptr + j);
                __m256 term = _mm256_mul_ps(lr_vec, grad_vec);
                data_vec = _mm256_sub_ps(data_vec, term);
                _mm256_storeu_ps(data_ptr + j, data_vec);

                // Zero gradients after update
                _mm256_storeu_ps(grad_ptr + j, _mm256_setzero_ps());
            }

            // Scalar fallback
            for (; j < num_elements; ++j)
            {
                data_ptr[j] -= lr * grad_ptr[j];
                grad_ptr[j] = 0.0f;
            }
        }
        // --- Non-Contiguous Path ---
        else
        {
            // Use separate index calculations for data and grad
            int* indices = calloc(params[i]->ndim, sizeof(int));
            if (!indices)
            {
                LOG_ERROR("sgd: Failed to allocate memory for indices for parameter %d.", i);
                continue;
            }

            for (int k = 0; k < num_elements; ++k)
            {
                size_t data_idx = get_flat_index(params[i], indices);
                // Assuming grad has same layout - if not, calculate separately
                params[i]->data->data[data_idx] -= lr * params[i]->grad->data->data[data_idx];
                params[i]->grad->data->data[data_idx] = 0.0f; // Zero gradient

                // Increment indices (your existing logic is fine)
                int carry = 1;
                for (int dim = params[i]->ndim - 1; dim >= 0 && carry; --dim)
                {
                    indices[dim]++;
                    if (indices[dim] < params[i]->shape[dim])
                    {
                        break;
                    }
                    else
                    {
                        indices[dim] = 0;
                        carry = 1;
                    }
                }
            }
            free(indices);
        }
    }
}

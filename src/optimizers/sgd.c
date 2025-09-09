#include "optimizers/optimizers.h"
#include "utils.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void sgd(Tensor **params, int num_params, float lr) {
  __m256 lr_vec = _mm256_set1_ps(lr);

  for (int i = 0; i < num_params; ++i) {
    if (!params[i] || !params[i]->requires_grad || !params[i]->grad ||
        !params[i]->grad->ptr || !params[i]->data || !params[i]->data->ptr) {
      IDRAK_WARNING("sgd: Skipping parameter %d due to invalid tensor, missing grad, or missing data.\n", i);
      continue;
    }

    int num_elements = numel(params[i]->shape, params[i]->ndim);
    if (num_elements == 0) {
      IDRAK_WARNING("sgd: Skipping parameter %d due to zero elements.\n", i);
      continue;
    }

    // --- Contiguous Path ---
    if (is_contiguous(params[i])) {
      float *data_ptr = params[i]->data->ptr;
      float *grad_ptr = params[i]->grad->ptr;

      int j = 0;
      for (; j + SIMD_WIDTH - 1 < num_elements; j += SIMD_WIDTH) {
        __m256 data_vec = _mm256_loadu_ps(data_ptr + j);
        __m256 grad_vec = _mm256_loadu_ps(grad_ptr + j);
        __m256 term = _mm256_mul_ps(lr_vec, grad_vec);
        data_vec = _mm256_sub_ps(data_vec, term);
        _mm256_storeu_ps(data_ptr + j, data_vec);

        // Zero gradients after update
        _mm256_storeu_ps(grad_ptr + j, _mm256_setzero_ps());
      }

      // Scalar fallback
      for (; j < num_elements; ++j) {
        data_ptr[j] -= lr * grad_ptr[j];
        grad_ptr[j] = 0.0f; // Zero gradient
      }
    }
    // --- Non-Contiguous Path ---
    else {
      // Use separate index calculations for data and grad
      int *indices = (int *)calloc(params[i]->ndim, sizeof(int));
      if (!indices) {
        IDRAK_ERROR("sgd: Failed to allocate memory for indices for parameter %d.\n", i);
        continue;
      }

      for (int k = 0; k < num_elements; ++k) {
        size_t data_idx = get_flat_index(params[i], indices);
        // Assuming grad has same layout - if not, calculate separately
        params[i]->data->ptr[data_idx] -= lr * params[i]->grad->ptr[data_idx];
        params[i]->grad->ptr[data_idx] = 0.0f; // Zero gradient

        // Increment indices (your existing logic is fine)
        int carry = 1;
        for (int dim = params[i]->ndim - 1; dim >= 0 && carry; --dim) {
          indices[dim] += carry;
          if (indices[dim] < params[i]->shape[dim]) {
            carry = 0;
          } else {
            indices[dim] = 0;
            carry = 1;
          }
        }
      }
      free(indices);
    }
  }
}

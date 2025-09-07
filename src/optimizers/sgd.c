#include "optimizers/optimizers.h"
#include "utils.h"
#include <immintrin.h>
#include <stdio.h>
#define SIMD_WIDTH 8

void sgd(Tensor **params, int num_params, float lr) {
  __m256 lr_vec = _mm256_set1_ps(lr);

  for (int i = 0; i < num_params; ++i) {
    // Robust checks for parameter validity
    if (!params[i]) {
      fprintf(stderr,
              "Warning: Parameter at index %d is NULL, skipping SGD update.\n",
              i);
      continue;
    }
    // Only update if gradient is required and gradient buffer exists
    if (!params[i]->requires_grad || !params[i]->grad ||
        !params[i]->grad->ptr) {
      // fprintf(stderr, "Info: Parameter %d does not require grad or has no
      // grad buffer, skipping.\n", i);
      continue;
    }

    int num_elements = numel(params[i]->shape, params[i]->ndim);
    // Further checks for data pointer validity
    if (num_elements == 0 || !params[i]->data || !params[i]->data->ptr) {
      fprintf(stderr,
              "Warning: Parameter at index %d has no elements or data pointer, "
              "skipping SGD update.\n",
              i);
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
      }

      // Scalar fallback for remaining elements
      for (; j < num_elements; ++j) {
        data_ptr[j] -= lr * grad_ptr[j];
      }
    }
    // --- Non-Contiguous Path ---
    else {
      int *current_indices = (int *)calloc(params[i]->ndim, sizeof(int));
      if (!current_indices) {
        fprintf(stderr,
                "Error: Failed to allocate current_indices for param %d, "
                "skipping SGD update for this param.\n",
                i);
        continue; // Skip this parameter if calloc fails
      }

      for (int k = 0; k < num_elements; ++k) {
        size_t data_flat_idx = get_flat_index(
            params[i], current_indices); // Use size_t for flat_idx

        // Perform the update
        params[i]->data->ptr[data_flat_idx] -=
            lr * params[i]->grad->ptr[data_flat_idx];

        // Increment current_indices for next element
        int carry = 1;
        for (int dim = params[i]->ndim - 1; dim >= 0 && carry; --dim) {
          current_indices[dim] += carry;
          if (current_indices[dim] < params[i]->shape[dim]) {
            carry = 0; // No carry to the next dimension
          } else {
            current_indices[dim] = 0; // Reset current dimension and carry over
            carry = 1;
          }
        }
      }
      free(current_indices); // Free allocated memory for current_indices
    }
  }
}

#include "optimizers/optimizers.h"
#include "utils.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void sgd(Tensor **params, int num_params, float lr) {
  __m256 lr_vec = _mm256_set1_ps(lr);

  for (int i = 0; i < num_params; ++i) {
    int num_elements = numel(params[i]->shape, params[i]->ndim);

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

      for (; j < num_elements; ++j) {
        data_ptr[j] -= lr * grad_ptr[j];
      }
    } else {
      int *current_indices = (int *)calloc(params[i]->ndim, sizeof(int));
      if (!current_indices) {
        continue;
      }

      for (int k = 0; k < num_elements; ++k) {
        int data_flat_idx = get_flat_index(params[i], current_indices);
        params[i]->data->ptr[data_flat_idx] -= lr * params[i]->grad->ptr[data_flat_idx];

        for (int dim = params[i]->ndim - 1; dim >= 0; --dim) {
          current_indices[dim]++;
          if (current_indices[dim] < params[i]->shape[dim]) {
            break;
          } else {
            current_indices[dim] = 0;
            if (dim == 0) {
              break;
            }
          }
        }
      }
      free(current_indices);
    }
  }
}

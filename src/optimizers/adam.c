#include "optimizers/optimizers.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

#ifndef SIMD_WIDTH
#define SIMD_WIDTH 8
#endif

void adam(Tensor **params, Tensor **m_estimates, Tensor **v_estimates,
          int num_params, int time_step, float learning_rate, float beta1,
          float beta2, float epsilon) {

  __m256 learning_rate_vec = _mm256_set1_ps(learning_rate);
  __m256 beta1_vec = _mm256_set1_ps(beta1);
  __m256 one_minus_beta1_vec = _mm256_set1_ps(1.0f - beta1);
  __m256 beta2_vec = _mm256_set1_ps(beta2);
  __m256 one_minus_beta2_vec = _mm256_set1_ps(1.0f - beta2);
  __m256 epsilon_vec = _mm256_set1_ps(epsilon);

  float beta1_pow_t = powf(beta1, time_step);
  float beta2_pow_t = powf(beta2, time_step);
  __m256 bias_correction_beta1_vec = _mm256_set1_ps(1.0f - beta1_pow_t);
  __m256 bias_correction_beta2_vec = _mm256_set1_ps(1.0f - beta2_pow_t);

  for (int i = 0; i < num_params; ++i) {
    int num_elements = numel(params[i]->shape, params[i]->ndim);

    float *current_m_estimates = m_estimates[i]->data;
    float *current_v_estimates = v_estimates[i]->data;

    if (is_contiguous(params[i])) {
      float *param_data = params[i]->data;
      float *param_grad = params[i]->grad;

      int j = 0;
      for (; j + SIMD_WIDTH - 1 < num_elements; j += SIMD_WIDTH) {
        __m256 current_param_data_vec = _mm256_loadu_ps(param_data + j);
        __m256 current_param_grad_vec = _mm256_loadu_ps(param_grad + j);
        __m256 m_vec = _mm256_loadu_ps(current_m_estimates + j);
        __m256 v_vec = _mm256_loadu_ps(current_v_estimates + j);

        m_vec = _mm256_add_ps(
            _mm256_mul_ps(beta1_vec, m_vec),
            _mm256_mul_ps(one_minus_beta1_vec, current_param_grad_vec));
        _mm256_storeu_ps(current_m_estimates + j, m_vec);

        __m256 grad_squared_vec =
            _mm256_mul_ps(current_param_grad_vec, current_param_grad_vec);
        v_vec =
            _mm256_add_ps(_mm256_mul_ps(beta2_vec, v_vec),
                          _mm256_mul_ps(one_minus_beta2_vec, grad_squared_vec));
        _mm256_storeu_ps(current_v_estimates + j, v_vec);

        __m256 m_hat_vec = _mm256_div_ps(m_vec, bias_correction_beta1_vec);

        __m256 v_hat_vec = _mm256_div_ps(v_vec, bias_correction_beta2_vec);

        __m256 sqrt_v_hat_vec = _mm256_sqrt_ps(v_hat_vec);
        __m256 denominator_vec = _mm256_add_ps(sqrt_v_hat_vec, epsilon_vec);
        __m256 update_term_vec = _mm256_div_ps(m_hat_vec, denominator_vec);
        update_term_vec = _mm256_mul_ps(learning_rate_vec, update_term_vec);

        current_param_data_vec =
            _mm256_sub_ps(current_param_data_vec, update_term_vec);
        _mm256_storeu_ps(param_data + j, current_param_data_vec);
      }

      for (; j < num_elements; ++j) {
        current_m_estimates[j] =
            beta1 * current_m_estimates[j] + (1.0f - beta1) * param_grad[j];

        current_v_estimates[j] =
            beta2 * current_v_estimates[j] +
            (1.0f - beta2) * (param_grad[j] * param_grad[j]);

        float m_hat = current_m_estimates[j] / (1.0f - beta1_pow_t);

        float v_hat = current_v_estimates[j] / (1.0f - beta2_pow_t);

        float update_term = learning_rate * m_hat / (sqrtf(v_hat) + epsilon);

        param_data[j] -= update_term;
      }
    } else {
      int *current_indices = (int *)calloc(params[i]->ndim, sizeof(int));
      if (!current_indices) {
        continue;
      }

      for (int k = 0; k < num_elements; ++k) {
        int flat_idx = get_flat_index(params[i], current_indices);

        current_m_estimates[flat_idx] =
            beta1 * current_m_estimates[flat_idx] +
            (1.0f - beta1) * params[i]->grad[flat_idx];

        current_v_estimates[flat_idx] =
            beta2 * current_v_estimates[flat_idx] +
            (1.0f - beta2) *
                (params[i]->grad[flat_idx] * params[i]->grad[flat_idx]);

        float m_hat = current_m_estimates[flat_idx] / (1.0f - beta1_pow_t);

        float v_hat = current_v_estimates[flat_idx] / (1.0f - beta2_pow_t);

        float update_term = learning_rate * m_hat / (sqrtf(v_hat) + epsilon);

        params[i]->data[flat_idx] -= update_term;

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

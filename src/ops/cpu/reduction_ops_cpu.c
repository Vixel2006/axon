#include "ops.h"
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sum_op(Tensor *a, Tensor *out, int axis, bool keepdim) {
  out->ndim = keepdim ? a->ndim : a->ndim - 1;

  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    free_tensor(out);
    return;
  }

  int j = 0;
  for (int i = 0; i < a->ndim; ++i) {
    if (i == axis) {
      if (keepdim) {
        out->shape[j++] = 1;
      }
    } else {
      out->shape[j++] = a->shape[i];
    }
  }

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides && out->ndim > 0) {
    fprintf(stderr, "Error: Failed to allocate memory for out->strides\n");
    free(out->shape);
    free_tensor(out);
    return;
  }

  size_t out_total_size = 1;
  for (int i = 0; i < out->ndim; ++i) {
    out_total_size *= out->shape[i];
  }
  out->data = (float *)malloc(out_total_size * sizeof(float));
  if (!out->data) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data\n");
    free(out->shape);
    free(out->strides);
    free_tensor(out);
    return;
  }

  memset(out->data, 0, out_total_size * sizeof(float));

  int batch_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch_num *= a->shape[i];
  }

  int reduction_size = a->shape[axis];

  int post_num = 1;
  for (int i = axis + 1; i < a->ndim; ++i) {
    post_num *= a->shape[i];
  }

  int *out_coords = (int *)malloc(out->ndim * sizeof(int));
  if (!out_coords) {
    fprintf(stderr, "Error: Failed to allocate memory for out_coords\n");
    free_tensor(out);
    return;
  }

  size_t reduction_stride_a = a->strides[axis];
  const int VEC_SIZE = 8;

  for (int batch_idx_linear = 0; batch_idx_linear < batch_num;
       ++batch_idx_linear) {
    for (int post_idx_linear = 0; post_idx_linear < post_num;
         ++post_idx_linear) {
      float current_sum_val = 0.0f;

      int temp_idx = batch_idx_linear;
      int out_dim_curr = 0;
      for (int d = 0; d < axis; ++d) {
        size_t product_trailing_prefix = 1;
        for (int k = d + 1; k < axis; ++k) {
          product_trailing_prefix *= a->shape[k];
        }
        out_coords[out_dim_curr++] = temp_idx / product_trailing_prefix;
        temp_idx %= product_trailing_prefix;
      }

      if (keepdim) {
        out_coords[out_dim_curr++] = 0;
      }

      temp_idx = post_idx_linear;
      for (int d = axis + 1; d < a->ndim; ++d) {
        size_t product_trailing_suffix = 1;
        for (int k = d + 1; k < a->ndim; ++k) {
          product_trailing_suffix *= a->shape[k];
        }
        out_coords[out_dim_curr++] = temp_idx / product_trailing_suffix;
        temp_idx %= product_trailing_suffix;
      }

      size_t out_offset = 0;
      for (int d = 0; d < out->ndim; ++d) {
        out_offset += (size_t)out_coords[d] * out->strides[d];
      }

      size_t base_in_offset = 0;
      int current_a_dim_for_base = 0;

      temp_idx = batch_idx_linear;
      for (int d = 0; d < axis; ++d) {
        size_t product_trailing_prefix = 1;
        for (int k = d + 1; k < axis; ++k) {
          product_trailing_prefix *= a->shape[k];
        }
        int coord = temp_idx / product_trailing_prefix;
        base_in_offset += (size_t)coord * a->strides[current_a_dim_for_base++];
        temp_idx %= product_trailing_prefix;
      }
      current_a_dim_for_base++;

      temp_idx = post_idx_linear;
      for (int d = axis + 1; d < a->ndim; ++d) {
        size_t product_trailing_suffix = 1;
        for (int k = d + 1; k < a->ndim; ++k) {
          product_trailing_suffix *= a->shape[k];
        }
        int coord = temp_idx / product_trailing_suffix;
        base_in_offset += (size_t)coord * a->strides[current_a_dim_for_base++];
        temp_idx %= product_trailing_suffix;
      }

      int i = 0;

      if (reduction_stride_a == 1) {
        __m256 sum_vec = _mm256_setzero_ps();

        for (; i + (VEC_SIZE - 1) < reduction_size; i += VEC_SIZE) {
          __m256 vec_a = _mm256_loadu_ps(&a->data[base_in_offset + (size_t)i]);
          sum_vec = _mm256_add_ps(sum_vec, vec_a);
        }

        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 lo_half = _mm256_extractf128_ps(sum_vec, 0);
        __m128 hi_half = _mm256_extractf128_ps(sum_vec, 1);
        __m128 total_sum_m128 = _mm_add_ps(lo_half, hi_half);
        current_sum_val = _mm_cvtss_f32(total_sum_m128);
      }

      for (; i < reduction_size; ++i) {
        current_sum_val +=
            a->data[base_in_offset + (size_t)i * reduction_stride_a];
      }

      out->data[out_offset] = current_sum_val;
    }
  }

  free(out_coords);

  out->requires_grad = a->requires_grad;
}

void mean_op(Tensor *a, Tensor *out, int axis, bool keepdim) {
  out->ndim = keepdim ? a->ndim : a->ndim - 1;

  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    free_tensor(out);
    return;
  }

  int j = 0;
  for (int i = 0; i < a->ndim; ++i) {
    if (i == axis) {
      if (keepdim) {
        out->shape[j++] = 1;
      }
    } else {
      out->shape[j++] = a->shape[i];
    }
  }

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides && out->ndim > 0) {
    fprintf(stderr, "Error: Failed to allocate memory for out->strides\n");
    free(out->shape);
    free_tensor(out);
    return;
  }

  size_t out_total_size = 1;
  for (int i = 0; i < out->ndim; ++i) {
    out_total_size *= out->shape[i];
  }
  out->data = (float *)malloc(out_total_size * sizeof(float));
  if (!out->data) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data\n");
    free(out->shape);
    free(out->strides);
    free_tensor(out);
    return;
  }

  memset(out->data, 0, out_total_size * sizeof(float));

  int batch_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch_num *= a->shape[i];
  }

  int reduction_size = a->shape[axis];

  int post_num = 1;
  for (int i = axis + 1; i < a->ndim; ++i) {
    post_num *= a->shape[i];
  }

  int *out_coords = (int *)malloc(out->ndim * sizeof(int));
  if (!out_coords) {
    fprintf(stderr, "Error: Failed to allocate memory for out_coords\n");
    free_tensor(out);
    return;
  }

  size_t reduction_stride_a = a->strides[axis];
  const int VEC_SIZE = 8;

  for (int batch_idx_linear = 0; batch_idx_linear < batch_num;
       ++batch_idx_linear) {
    for (int post_idx_linear = 0; post_idx_linear < post_num;
         ++post_idx_linear) {
      float current_sum_val = 0.0f;

      int temp_idx = batch_idx_linear;
      int out_dim_curr = 0;
      for (int d = 0; d < axis; ++d) {
        size_t product_trailing_prefix = 1;
        for (int k = d + 1; k < axis; ++k) {
          product_trailing_prefix *= a->shape[k];
        }
        out_coords[out_dim_curr++] = temp_idx / product_trailing_prefix;
        temp_idx %= product_trailing_prefix;
      }

      if (keepdim) {
        out_coords[out_dim_curr++] = 0;
      }

      temp_idx = post_idx_linear;
      for (int d = axis + 1; d < a->ndim; ++d) {
        size_t product_trailing_suffix = 1;
        for (int k = d + 1; k < a->ndim; ++k) {
          product_trailing_suffix *= a->shape[k];
        }
        out_coords[out_dim_curr++] = temp_idx / product_trailing_suffix;
        temp_idx %= product_trailing_suffix;
      }

      size_t out_offset = 0;
      for (int d = 0; d < out->ndim; ++d) {
        out_offset += (size_t)out_coords[d] * out->strides[d];
      }

      size_t base_in_offset = 0;
      int current_a_dim_for_base = 0;

      temp_idx = batch_idx_linear;
      for (int d = 0; d < axis; ++d) {
        size_t product_trailing_prefix = 1;
        for (int k = d + 1; k < axis; ++k) {
          product_trailing_prefix *= a->shape[k];
        }
        int coord = temp_idx / product_trailing_prefix;
        base_in_offset += (size_t)coord * a->strides[current_a_dim_for_base++];
        temp_idx %= product_trailing_prefix;
      }
      current_a_dim_for_base++;

      temp_idx = post_idx_linear;
      for (int d = axis + 1; d < a->ndim; ++d) {
        size_t product_trailing_suffix = 1;
        for (int k = d + 1; k < a->ndim; ++k) {
          product_trailing_suffix *= a->shape[k];
        }
        int coord = temp_idx / product_trailing_suffix;
        base_in_offset += (size_t)coord * a->strides[current_a_dim_for_base++];
        temp_idx %= product_trailing_suffix;
      }

      int i = 0;

      if (reduction_stride_a == 1) {
        __m256 sum_vec = _mm256_setzero_ps();

        for (; i + (VEC_SIZE - 1) < reduction_size; i += VEC_SIZE) {
          __m256 vec_a = _mm256_loadu_ps(&a->data[base_in_offset + (size_t)i]);
          sum_vec = _mm256_add_ps(sum_vec, vec_a);
        }

        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 lo_half = _mm256_extractf128_ps(sum_vec, 0);
        __m128 hi_half = _mm256_extractf128_ps(sum_vec, 1);
        __m128 total_sum_m128 = _mm_add_ps(lo_half, hi_half);
        current_sum_val = _mm_cvtss_f32(total_sum_m128);
      }

      for (; i < reduction_size; ++i) {
        current_sum_val +=
            a->data[base_in_offset + (size_t)i * reduction_stride_a];
      }

      out->data[out_offset] = current_sum_val / reduction_size;
    }
  }

  free(out_coords);

  out->requires_grad = a->requires_grad;
}

void max_op(Tensor *a, Tensor *out, int axis, bool keepdim) {
  out->ndim = keepdim ? a->ndim : a->ndim - 1;

  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    free_tensor(out);
    return;
  }

  int j = 0;
  for (int i = 0; i < a->ndim; ++i) {
    if (i == axis) {
      if (keepdim) {
        out->shape[j++] = 1;
      }
    } else {
      out->shape[j++] = a->shape[i];
    }
  }

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides && out->ndim > 0) {
    fprintf(stderr, "Error: Failed to allocate memory for out->strides\n");
    free(out->shape);
    free_tensor(out);
    return;
  }

  size_t out_total_size = 1;
  for (int i = 0; i < out->ndim; ++i) {
    out_total_size *= out->shape[i];
  }
  out->data = (float *)malloc(out_total_size * sizeof(float));
  if (!out->data) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data\n");
    free(out->shape);
    free(out->strides);
    free_tensor(out);
    return;
  }

  for (size_t k = 0; k < out_total_size; ++k) {
    out->data[k] = -FLT_MAX;
  }

  int batch_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch_num *= a->shape[i];
  }

  int reduction_size = a->shape[axis];

  int post_num = 1;
  for (int i = axis + 1; i < a->ndim; ++i) {
    post_num *= a->shape[i];
  }

  int *out_coords = (int *)malloc(out->ndim * sizeof(int));
  if (!out_coords) {
    fprintf(stderr, "Error: Failed to allocate memory for out_coords\n");
    free_tensor(out);
    return;
  }

  size_t reduction_stride_a = a->strides[axis];
  const int VEC_SIZE = 8;
  const __m256 neg_flt_max_vec = _mm256_set1_ps(-FLT_MAX);

  for (int batch_idx_linear = 0; batch_idx_linear < batch_num;
       ++batch_idx_linear) {
    for (int post_idx_linear = 0; post_idx_linear < post_num;
         ++post_idx_linear) {
      float current_max_val = -FLT_MAX;

      int temp_idx = batch_idx_linear;
      int out_dim_curr = 0;
      for (int d = 0; d < axis; ++d) {
        size_t product_trailing_prefix = 1;
        for (int k = d + 1; k < axis; ++k) {
          product_trailing_prefix *= a->shape[k];
        }
        out_coords[out_dim_curr++] = temp_idx / product_trailing_prefix;
        temp_idx %= product_trailing_prefix;
      }

      if (keepdim) {
        out_coords[out_dim_curr++] = 0;
      }

      temp_idx = post_idx_linear;
      for (int d = axis + 1; d < a->ndim; ++d) {
        size_t product_trailing_suffix = 1;
        for (int k = d + 1; k < a->ndim; ++k) {
          product_trailing_suffix *= a->shape[k];
        }
        out_coords[out_dim_curr++] = temp_idx / product_trailing_suffix;
        temp_idx %= product_trailing_suffix;
      }

      size_t out_offset = 0;
      for (int d = 0; d < out->ndim; ++d) {
        out_offset += (size_t)out_coords[d] * out->strides[d];
      }

      size_t base_in_offset = 0;
      int current_a_dim_for_base = 0;

      temp_idx = batch_idx_linear;
      for (int d = 0; d < axis; ++d) {
        size_t product_trailing_prefix = 1;
        for (int k = d + 1; k < axis; ++k) {
          product_trailing_prefix *= a->shape[k];
        }
        int coord = temp_idx / product_trailing_prefix;
        base_in_offset += (size_t)coord * a->strides[current_a_dim_for_base++];
        temp_idx %= product_trailing_prefix;
      }
      current_a_dim_for_base++;

      temp_idx = post_idx_linear;
      for (int d = axis + 1; d < a->ndim; ++d) {
        size_t product_trailing_suffix = 1;
        for (int k = d + 1; k < a->ndim; ++k) {
          product_trailing_suffix *= a->shape[k];
        }
        int coord = temp_idx / product_trailing_suffix;
        base_in_offset += (size_t)coord * a->strides[current_a_dim_for_base++];
        temp_idx %= product_trailing_suffix;
      }

      int i = 0;

      if (reduction_stride_a == 1) {
        __m256 max_vec = neg_flt_max_vec;

        for (; i + (VEC_SIZE - 1) < reduction_size; i += VEC_SIZE) {
          __m256 vec_a = _mm256_loadu_ps(&a->data[base_in_offset + (size_t)i]);
          max_vec = _mm256_max_ps(max_vec, vec_a);
        }

        __m128 vlow = _mm256_castps256_ps128(max_vec);
        __m128 vhigh = _mm256_extractf128_ps(max_vec, 1);
        vlow = _mm_max_ps(vlow, vhigh);
        vlow = _mm_max_ps(vlow,
                          _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1)));
        vlow = _mm_max_ps(vlow,
                          _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2)));
        current_max_val = _mm_cvtss_f32(vlow);
      }

      for (; i < reduction_size; ++i) {
        current_max_val =
            fmaxf(current_max_val,
                  a->data[base_in_offset + (size_t)i * reduction_stride_a]);
      }

      out->data[out_offset] = current_max_val;
    }
  }

  free(out_coords);

  out->requires_grad = a->requires_grad;
}

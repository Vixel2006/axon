#include "utils.h"
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "ops/ops.h"

void sum_op(Tensor *a, Tensor *out, int axis, bool keepdim) {
  DEBUG_PRINT("sum_op: Performing sum reduction along axis %d (keepdim=%d)\n",
              axis, keepdim);

  if (out->shape) {
    free(out->shape);
    out->shape = NULL;
  }
  if (out->strides) {
    free(out->strides);
    out->strides = NULL;
  }
  if (out->data && out->data->ptr) {
    free(out->data->ptr);
    out->data->ptr = NULL;
  }

  // 1. Adjust ndim for output (drop or keep reduction axis)
  out->ndim = keepdim ? a->ndim : a->ndim - 1;

  // 2. Build output shape (drop axis unless keepdim=true)
  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    free_tensor(&out);
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
    free_tensor(&out);
    return;
  }

  size_t out_total_size = 1;
  for (int i = 0; i < out->ndim; ++i) {
    out_total_size *= out->shape[i];
  }
  out->data->ptr = (float *)malloc(out_total_size * sizeof(float));
  if (!out->data->ptr) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data->ptr\n");
    free(out->shape);
    free(out->strides);
    free_tensor(&out);
    return;
  }

  memset(out->data->ptr, 0, out_total_size * sizeof(float));

  // 3. Decompose shape into batch × reduction × post parts
  int batch_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch_num *= a->shape[i];
  }

  int reduction_size = a->shape[axis];

  int post_num = 1;
  for (int i = axis + 1; i < a->ndim; ++i) {
    post_num *= a->shape[i];
  }

  // Temp buffer for computing output coordinates
  int *out_coords = (int *)malloc(out->ndim * sizeof(int));
  if (!out_coords) {
    fprintf(stderr, "Error: Failed to allocate memory for out_coords\n");
    free_tensor(&out);
    return;
  }

  size_t reduction_stride_a = a->strides[axis];
  const int VEC_SIZE = 8;

  // 4. Loop over "batches" × "post" slices (everything except reduced axis)
  for (int batch_idx_linear = 0; batch_idx_linear < batch_num;
       ++batch_idx_linear) {
    for (int post_idx_linear = 0; post_idx_linear < post_num;
         ++post_idx_linear) {
      float current_sum_val = 0.0f;

      // ========================
      // Compute output coords
      // ========================
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

      // Map output coords to flat index in out->data->ptr
      size_t out_offset = 0;
      for (int d = 0; d < out->ndim; ++d) {
        out_offset += (size_t)out_coords[d] * out->strides[d];
      }

      // =========================
      // Compute base input offset
      // =========================
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
      current_a_dim_for_base++; // skip reduction axis

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

      // =======================
      // Reduction loop
      // =======================
      int i = 0;

      if (reduction_stride_a == 1) {
        __m256 sum_vec = _mm256_setzero_ps();

        for (; i + (VEC_SIZE - 1) < reduction_size; i += VEC_SIZE) {
          __m256 vec_a =
              _mm256_loadu_ps(&a->data->ptr[base_in_offset + (size_t)i]);
          sum_vec = _mm256_add_ps(sum_vec, vec_a);
        }

        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 lo_half = _mm256_extractf128_ps(sum_vec, 0);
        __m128 hi_half = _mm256_extractf128_ps(sum_vec, 1);
        __m128 total_sum_m128 = _mm_add_ps(lo_half, hi_half);
        current_sum_val = _mm_cvtss_f32(total_sum_m128);

        for (; i < reduction_size; ++i) {
          current_sum_val +=
              a->data->ptr[base_in_offset + (size_t)i * reduction_stride_a];
        }
      } else {
        for (i = 0; i < reduction_size; ++i) {
          current_sum_val +=
              a->data->ptr[base_in_offset + (size_t)i * reduction_stride_a];
        }
      }

      out->data->ptr[out_offset] = current_sum_val;
    }
  }

  free(out_coords);

  out->requires_grad = a->requires_grad;
}

void mean_op(Tensor *a, Tensor *out, int axis, bool keepdim) {
  DEBUG_PRINT("mean_op: Performing mean reduction along axis %d (keepdim=%d)\n",
              axis, keepdim);

  if (out->shape) {
    free(out->shape);
    out->shape = NULL;
  }
  if (out->strides) {
    free(out->strides);
    out->strides = NULL;
  }
  if (out->data && out->data->ptr) {
    free(out->data->ptr);
    out->data->ptr = NULL;
  }

  out->ndim = keepdim ? a->ndim : a->ndim - 1;

  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    free_tensor(&out);
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
    free_tensor(&out);
    return;
  }

  size_t out_total_size = 1;
  for (int i = 0; i < out->ndim; ++i) {
    out_total_size *= out->shape[i];
  }
  out->data->ptr = (float *)malloc(out_total_size * sizeof(float));
  if (!out->data->ptr) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data->ptr\n");
    free(out->shape);
    free(out->strides);
    free_tensor(&out);
    return;
  }

  memset(out->data->ptr, 0, out_total_size * sizeof(float));

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
    free_tensor(&out);
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
          __m256 vec_a =
              _mm256_loadu_ps(&a->data->ptr[base_in_offset + (size_t)i]);
          sum_vec = _mm256_add_ps(sum_vec, vec_a);
        }

        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 lo_half = _mm256_extractf128_ps(sum_vec, 0);
        __m128 hi_half = _mm256_extractf128_ps(sum_vec, 1);
        __m128 total_sum_m128 = _mm_add_ps(lo_half, hi_half);
        current_sum_val = _mm_cvtss_f32(total_sum_m128);

        for (; i < reduction_size; ++i) {
          current_sum_val +=
              a->data->ptr[base_in_offset + (size_t)i * reduction_stride_a];
        }
      } else {
        for (i = 0; i < reduction_size; ++i) { // Reset i for this loop
          current_sum_val +=
              a->data->ptr[base_in_offset + (size_t)i * reduction_stride_a];
        }
      }

      out->data->ptr[out_offset] = current_sum_val / reduction_size;
    }
  }

  free(out_coords);

  out->requires_grad = a->requires_grad;
}

void max_op(Tensor *a, Tensor *out, int axis, bool keepdim) {
  DEBUG_PRINT("max_op: Performing max reduction along axis %d (keepdim=%d)\n",
              axis, keepdim);

  if (out->shape) {
    free(out->shape);
    out->shape = NULL;
  }
  if (out->strides) {
    free(out->strides);
    out->strides = NULL;
  }
  if (out->data && out->data->ptr) {
    free(out->data->ptr);
    out->data->ptr = NULL;
  }

  out->ndim = keepdim ? a->ndim : a->ndim - 1;

  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    free_tensor(&out);
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
    free_tensor(&out);
    return;
  }

  size_t out_total_size = 1;
  for (int i = 0; i < out->ndim; ++i) {
    out_total_size *= out->shape[i];
  }
  out->data->ptr = (float *)malloc(out_total_size * sizeof(float));
  if (!out->data->ptr) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data->ptr\n");
    free(out->shape);
    free(out->strides);
    free_tensor(&out);
    return;
  }

  for (size_t k = 0; k < out_total_size; ++k) {
    out->data->ptr[k] = -FLT_MAX;
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
    free_tensor(&out);
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
          __m256 vec_a =
              _mm256_loadu_ps(&a->data->ptr[base_in_offset + (size_t)i]);
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

        for (; i < reduction_size; ++i) {
          current_max_val = fmaxf(
              current_max_val,
              a->data->ptr[base_in_offset + (size_t)i * reduction_stride_a]);
        }
      } else {
        for (i = 0; i < reduction_size; ++i) { // Reset i for this loop
          current_max_val = fmaxf(
              current_max_val,
              a->data->ptr[base_in_offset + (size_t)i * reduction_stride_a]);
        }
      }

      out->data->ptr[out_offset] = current_max_val;
    }
  }

  free(out_coords);

  out->requires_grad = a->requires_grad;
}

void sum_full_op(Tensor *a, Tensor *out) {
  DEBUG_PRINT("sum_full_op: Performing full sum reduction\n");

  if (out && out->shape) {
    free(out->shape);
    out->shape = NULL;
  }
  if (out && out->strides) {
    free(out->strides);
    out->strides = NULL;
  }
  if (out && out->data && out->data->ptr) {
    free(out->data->ptr);
    out->data->ptr = NULL;
  }

  out->ndim = 0;
  out->shape = NULL;
  out->strides = NULL;

  out->data->ptr = (float *)malloc(sizeof(float));
  if (!out->data->ptr) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data->ptr\n");
    return;
  }

  size_t total_elements = 1;
  for (int i = 0; i < a->ndim; ++i) {
    total_elements *= a->shape[i];
  }

  bool is_contiguous = true;
  if (a->ndim > 0) {
    size_t expected_stride = 1;
    for (int i = a->ndim - 1; i >= 0; --i) {
      if (a->strides[i] != expected_stride) {
        is_contiguous = false;
        break;
      }
      expected_stride *= a->shape[i];
    }
  }

  float total_sum = 0.0f;
  const int VEC_SIZE = 8;

  if (is_contiguous) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + (VEC_SIZE - 1) < total_elements; i += VEC_SIZE) {
      __m256 vec_a = _mm256_loadu_ps(&a->data->ptr[i]);
      sum_vec = _mm256_add_ps(sum_vec, vec_a);
    }

    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    __m128 lo_half = _mm256_extractf128_ps(sum_vec, 0);
    __m128 hi_half = _mm256_extractf128_ps(sum_vec, 1);
    __m128 total_sum_m128 = _mm_add_ps(lo_half, hi_half);
    total_sum = _mm_cvtss_f32(total_sum_m128);

    for (; i < total_elements; ++i) {
      total_sum += a->data->ptr[i];
    }
  } else {
    int *coords = (int *)malloc(a->ndim * sizeof(int));
    if (!coords) {
      fprintf(stderr, "Error: Failed to allocate memory for coords\n");
      free(out->data->ptr);
      out->data->ptr = NULL;
      return;
    }

    memset(coords, 0, a->ndim * sizeof(int));

    for (size_t elem = 0; elem < total_elements; ++elem) {
      size_t flat_idx = 0;
      for (int d = 0; d < a->ndim; ++d) {
        flat_idx += (size_t)coords[d] * a->strides[d];
      }

      total_sum += a->data->ptr[flat_idx];

      int carry = 1;
      for (int d = a->ndim - 1; d >= 0 && carry; --d) {
        coords[d] += carry;
        if (coords[d] < a->shape[d]) {
          carry = 0;
        } else {
          coords[d] = 0;
          carry = 1;
        }
      }
    }

    free(coords);
  }

  out->data->ptr[0] = total_sum;
  out->requires_grad = a->requires_grad;
}

void mean_full_op(Tensor *a, Tensor *out) {
  DEBUG_PRINT("mean_full_op: Performing full mean reduction\n");

  sum_full_op(a, out);

  if (out && out->data && out->data->ptr == NULL) {
    fprintf(stderr, "Error: sum_full_op failed, cannot compute mean.\n");
    return;
  }

  size_t total_elements = 1;
  for (int i = 0; i < a->ndim; ++i) {
    total_elements *= a->shape[i];
  }

  if (total_elements > 0) {
    out->data->ptr[0] /= total_elements;
  } else {
    if (out && out->data && out->data->ptr) {
      out->data->ptr[0] = 0.0f;
    }
  }
}

void max_full_op(Tensor *a, Tensor *out) {
  DEBUG_PRINT("max_full_op: Performing full max reduction\n");

  if (out && out->shape) {
    free(out->shape);
    out->shape = NULL;
  }
  if (out && out->strides) {
    free(out->strides);
    out->strides = NULL;
  }
  if (out && out->data && out->data->ptr) {
    free(out->data->ptr);
    out->data->ptr = NULL;
  }

  out->ndim = 0;
  out->shape = NULL;
  out->strides = NULL;

  out->data->ptr = (float *)malloc(sizeof(float));
  if (!out->data->ptr) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data->ptr\n");
    return;
  }

  size_t total_elements = 1;
  for (int i = 0; i < a->ndim; ++i) {
    total_elements *= a->shape[i];
  }

  if (total_elements == 0) {
    out->data->ptr[0] = -FLT_MAX;
    out->requires_grad = a->requires_grad;
    return;
  }

  bool is_contiguous = true;
  if (a->ndim > 0) {
    size_t expected_stride = 1;
    for (int i = a->ndim - 1; i >= 0; --i) {
      if (a->strides[i] != expected_stride) {
        is_contiguous = false;
        break;
      }
      expected_stride *= a->shape[i];
    }
  }

  float max_val = -FLT_MAX;
  const int VEC_SIZE = 8;

  if (is_contiguous) {
    __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
    size_t i = 0;

    for (; i + (VEC_SIZE - 1) < total_elements; i += VEC_SIZE) {
      __m256 vec_a = _mm256_loadu_ps(&a->data->ptr[i]);
      max_vec = _mm256_max_ps(max_vec, vec_a);
    }

    __m128 vlow = _mm256_castps256_ps128(max_vec);
    __m128 vhigh = _mm256_extractf128_ps(max_vec, 1);
    vlow = _mm_max_ps(vlow, vhigh);
    vlow =
        _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1)));
    vlow =
        _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2)));
    max_val = _mm_cvtss_f32(vlow);

    for (; i < total_elements; ++i) {
      max_val = fmaxf(max_val, a->data->ptr[i]);
    }
  } else {
    int *coords = (int *)malloc(a->ndim * sizeof(int));
    if (!coords) {
      fprintf(stderr, "Error: Failed to allocate memory for coords\n");
      free(out->data->ptr);
      out->data->ptr = NULL;
      return;
    }

    memset(coords, 0, a->ndim * sizeof(int));
    bool first = true;

    for (size_t elem = 0; elem < total_elements; ++elem) {
      size_t flat_idx = 0;
      for (int d = 0; d < a->ndim; ++d) {
        flat_idx += (size_t)coords[d] * a->strides[d];
      }

      if (first) {
        max_val = a->data->ptr[flat_idx];
        first = false;
      } else {
        max_val = fmaxf(max_val, a->data->ptr[flat_idx]);
      }

      int carry = 1;
      for (int d = a->ndim - 1; d >= 0 && carry; --d) {
        coords[d] += carry;
        if (coords[d] < a->shape[d]) {
          carry = 0;
        } else {
          coords[d] = 0;
          carry = 1;
        }
      }
    }

    free(coords);
  }

  out->data->ptr[0] = max_val;
  out->requires_grad = a->requires_grad;
}

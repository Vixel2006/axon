#include <immintrin.h>
#include <stdlib.h>

#include "autograd/autograd.h"
#include "autograd/autograd_utils.h"
#include "utils.h"

#define SIMD_WIDTH 8

void sum_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  if (!in->requires_grad) {
    return;
  }

  int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

  if (reduced_dim == -1) {
    // Handle case where no reduction happened (in and out have same shape)
    int size = numel(in->shape, in->ndim);
    if (!is_contiguous(in) || !is_contiguous(out)) {
      int *in_strides = in->strides;
      int *out_strides = out->strides;
      int *shape = in->shape;
      int ndim = in->ndim;

      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int in_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          in_offset += coord * in_strides[d];
          out_offset += coord * out_strides[d];
        }
        in->grad[in_offset] += out->grad[out_offset];
      }
    } else {
      int i = 0;
      for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
        __m256 in_grad = _mm256_loadu_ps(in->grad + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 new_in_grad = _mm256_add_ps(in_grad, dout);
        _mm256_storeu_ps(in->grad + i, new_in_grad);
      }
      for (; i < size; ++i) {
        in->grad[i] += out->grad[i];
      }
    }
    return;
  }

  // Reduction case
  int in_size = numel(in->shape, in->ndim);
  int *in_strides = in->strides;
  int *in_shape = in->shape;
  int in_ndim = in->ndim;

  int *out_strides = out->strides;
  int *out_shape = out->shape;
  int out_ndim = out->ndim;

  if (!is_contiguous(in) || !is_contiguous(out)) {
    // Non-contiguous path
    for (int in_linear_idx = 0; in_linear_idx < in_size; ++in_linear_idx) {
      int temp_in_linear_idx = in_linear_idx;
      int in_offset = 0;
      int out_offset = 0;
      int current_out_dim_idx = 0;

      for (int d = in_ndim - 1; d >= 0; --d) {
        int coord = temp_in_linear_idx % in_shape[d];
        temp_in_linear_idx /= in_shape[d];
        in_offset += coord * in_strides[d];

        if (d != reduced_dim) {
          out_offset += coord * out_strides[current_out_dim_idx];
          current_out_dim_idx++;
        }
      }
      in->grad[in_offset] += out->grad[out_offset];
    }
  } else {
    // Contiguous path (existing code)
    int reduce_size = in->shape[reduced_dim];
    int num_batches = numel(out->shape, out->ndim);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      float grad = out->grad[batch_idx];

      __m256 grad_vec = _mm256_set1_ps(grad);

      int base_offset = batch_idx * reduce_size;

      int i = 0;
      for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH) {
        __m256 data_vec = _mm256_loadu_ps(&in->grad[base_offset + i]);
        data_vec = _mm256_add_ps(data_vec, grad_vec);
        _mm256_storeu_ps(&in->grad[base_offset + i], data_vec);
      }

      for (; i < reduce_size; ++i) {
        in->grad[base_offset + i] += grad;
      }
    }
  }
}

void mean_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  if (!in->requires_grad) {
    return;
  }

  int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

  if (reduced_dim == -1) {
    // Handle case where no reduction happened (in and out have same shape)
    int size = numel(in->shape, in->ndim);
    if (!is_contiguous(in) || !is_contiguous(out)) {
      int *in_strides = in->strides;
      int *out_strides = out->strides;
      int *shape = in->shape;
      int ndim = in->ndim;

      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int in_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          in_offset += coord * in_strides[d];
          out_offset += coord * out_strides[d];
        }
        in->grad[in_offset] += out->grad[out_offset];
      }
    } else {
      int i = 0;
      for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
        __m256 in_grad = _mm256_loadu_ps(in->grad + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 new_in_grad = _mm256_add_ps(in_grad, dout);
        _mm256_storeu_ps(in->grad + i, new_in_grad);
      }
      for (; i < size; ++i) {
        in->grad[i] += out->grad[i];
      }
    }
    return;
  }

  // Reduction case
  int in_size = numel(in->shape, in->ndim);
  int *in_strides = in->strides;
  int *in_shape = in->shape;
  int in_ndim = in->ndim;

  int *out_strides = out->strides;
  int *out_shape = out->shape;
  int out_ndim = out->ndim;

  int reduce_size = in->shape[reduced_dim];

  if (!is_contiguous(in) || !is_contiguous(out)) {
    // Non-contiguous path
    for (int in_linear_idx = 0; in_linear_idx < in_size; ++in_linear_idx) {
      int temp_in_linear_idx = in_linear_idx;
      int in_offset = 0;
      int out_offset = 0;
      int current_out_dim_idx = 0;

      for (int d = in_ndim - 1; d >= 0; --d) {
        int coord = temp_in_linear_idx % in_shape[d];
        temp_in_linear_idx /= in_shape[d];
        in_offset += coord * in_strides[d];

        if (d != reduced_dim) {
          out_offset += coord * out_strides[current_out_dim_idx];
          current_out_dim_idx++;
        }
      }
      in->grad[in_offset] += out->grad[out_offset] / reduce_size;
    }
  } else {
    // Contiguous path (existing code)
    int num_batches = numel(out->shape, out->ndim);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      float grad = out->grad[batch_idx];

      __m256 grad_vec = _mm256_set1_ps(grad / reduce_size);

      int base_offset = batch_idx * reduce_size;

      int i = 0;
      for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH) {
        __m256 data_vec = _mm256_loadu_ps(&in->grad[base_offset + i]);
        data_vec = _mm256_add_ps(data_vec, grad_vec);
        _mm256_storeu_ps(&in->grad[base_offset + i], data_vec);
      }

      for (; i < reduce_size; ++i) {
        in->grad[base_offset + i] += grad / reduce_size;
      }
    }
  }
}

void max_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  if (!in->requires_grad) {
    return;
  }

  int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

  if (reduced_dim == -1) {
    // Handle case where no reduction happened (in and out have same shape)
    int size = numel(in->shape, in->ndim);
    if (!is_contiguous(in) || !is_contiguous(out)) {
      int *in_strides = in->strides;
      int *out_strides = out->strides;
      int *shape = in->shape;
      int ndim = in->ndim;

      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int in_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          in_offset += coord * in_strides[d];
          out_offset += coord * out_strides[d];
        }
        in->grad[in_offset] += out->grad[out_offset];
      }
    } else {
      int i = 0;
      for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
        __m256 in_grad = _mm256_loadu_ps(in->grad + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 new_in_grad = _mm256_add_ps(in_grad, dout);
        _mm256_storeu_ps(in->grad + i, new_in_grad);
      }
      for (; i < size; ++i) {
        in->grad[i] += out->grad[i];
      }
    }
    return;
  }

  // Reduction case
  int in_size = numel(in->shape, in->ndim);
  int *in_strides = in->strides;
  int *in_shape = in->shape;
  int in_ndim = in->ndim;

  int *out_strides = out->strides;
  int *out_shape = out->shape;
  int out_ndim = out->ndim;

  if (!is_contiguous(in) || !is_contiguous(out)) {
    // Non-contiguous path
    for (int in_linear_idx = 0; in_linear_idx < in_size; ++in_linear_idx) {
      int temp_in_linear_idx = in_linear_idx;
      int in_offset = 0;
      int out_offset = 0;
      int current_out_dim_idx = 0;

      for (int d = in_ndim - 1; d >= 0; --d) {
        int coord = temp_in_linear_idx % in_shape[d];
        temp_in_linear_idx /= in_shape[d];
        in_offset += coord * in_strides[d];

        if (d != reduced_dim) {
          out_offset += coord * out_strides[current_out_dim_idx];
          current_out_dim_idx++;
        }
      }

      if (in->data[in_offset] == out->data[out_offset]) {
        in->grad[in_offset] += out->grad[out_offset];
      }
    }
  } else {
    // Contiguous path (existing code with bug fix)
    int reduce_size = in->shape[reduced_dim];
    int num_batches = numel(out->shape, out->ndim);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      float grad = out->grad[batch_idx];
      float max = out->data[batch_idx];

      __m256 grad_vec = _mm256_set1_ps(grad);
      __m256 max_vec = _mm256_set1_ps(max);

      int base_offset = batch_idx * reduce_size;

      int i = 0;
      for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH) {
        __m256 data_vec = _mm256_loadu_ps(&in->data[base_offset + i]);
        __m256 mask = _mm256_cmp_ps(data_vec, max_vec, _CMP_EQ_OQ);
        __m256 grad_contrib = _mm256_and_ps(grad_vec, mask);
        __m256 in_grad = _mm256_loadu_ps(&in->grad[base_offset + i]);
        __m256 new_grad = _mm256_add_ps(in_grad, grad_contrib);
        _mm256_storeu_ps(&in->grad[base_offset + i], new_grad);
      }

      for (; i < reduce_size; ++i) {
        if (in->data[base_offset + i] == max) {
          in->grad[base_offset + i] += grad;
        }
      }
    }
  }
}

void sum_full_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  if (!in->requires_grad) {
    return;
  }

  // out is a scalar, so out->grad[0] contains the gradient
  float output_grad = out->grad[0];
  int in_size = numel(in->shape, in->ndim);

  if (is_contiguous(in)) {
    // Fast path: contiguous input tensor
    __m256 grad_vec = _mm256_set1_ps(output_grad);

    int i = 0;
    for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH) {
      __m256 in_grad = _mm256_loadu_ps(&in->grad[i]);
      __m256 new_grad = _mm256_add_ps(in_grad, grad_vec);
      _mm256_storeu_ps(&in->grad[i], new_grad);
    }

    // Handle remaining elements
    for (; i < in_size; ++i) {
      in->grad[i] += output_grad;
    }
  } else {
    // Slow path: non-contiguous input tensor
    int *in_strides = in->strides;
    int *in_shape = in->shape;
    int in_ndim = in->ndim;

    for (int linear_idx = 0; linear_idx < in_size; ++linear_idx) {
      int temp_idx = linear_idx;
      int in_offset = 0;

      // Convert linear index to multidimensional coordinates and compute offset
      for (int d = in_ndim - 1; d >= 0; --d) {
        int coord = temp_idx % in_shape[d];
        temp_idx /= in_shape[d];
        in_offset += coord * in_strides[d];
      }

      in->grad[in_offset] += output_grad;
    }
  }
}

/**
 * @brief Backward pass for mean_full_op.
 *
 * For full mean reduction, the gradient is distributed equally to all input
 * elements, but scaled by 1/N where N is the total number of elements.
 *
 * @param out    Output tensor (scalar from forward pass).
 * @param prev   Array of input tensors (prev[0] is the input tensor).
 * @param n_prev Number of input tensors (should be 1).
 * @param extras Additional data (unused).
 */
void mean_full_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  if (!in->requires_grad) {
    return;
  }

  float output_grad = out->grad[0];
  int in_size = numel(in->shape, in->ndim);

  float scaled_grad = output_grad / in_size;

  if (is_contiguous(in)) {
    __m256 grad_vec = _mm256_set1_ps(scaled_grad);

    int i = 0;
    for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH) {
      __m256 in_grad = _mm256_loadu_ps(&in->grad[i]);
      __m256 new_grad = _mm256_add_ps(in_grad, grad_vec);
      _mm256_storeu_ps(&in->grad[i], new_grad);
    }

    for (; i < in_size; ++i) {
      in->grad[i] += scaled_grad;
    }
  } else {
    int *in_strides = in->strides;
    int *in_shape = in->shape;
    int in_ndim = in->ndim;

    for (int linear_idx = 0; linear_idx < in_size; ++linear_idx) {
      int temp_idx = linear_idx;
      int in_offset = 0;

      for (int d = in_ndim - 1; d >= 0; --d) {
        int coord = temp_idx % in_shape[d];
        temp_idx /= in_shape[d];
        in_offset += coord * in_strides[d];
      }

      in->grad[in_offset] += scaled_grad;
    }
  }
}

/**
 * @brief Backward pass for max_full_op.
 *
 * For full max reduction, gradients only flow to input elements that achieved
 * the maximum value. If multiple elements have the same maximum value,
 * the gradient is distributed among them.
 *
 * @param out    Output tensor (scalar from forward pass).
 * @param prev   Array of input tensors (prev[0] is the input tensor).
 * @param n_prev Number of input tensors (should be 1).
 * @param extras Additional data (unused).
 */
void max_full_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  if (!in->requires_grad) {
    return;
  }

  float output_grad = out->grad[0];
  float max_val = out->data[0];
  int in_size = numel(in->shape, in->ndim);

  if (is_contiguous(in)) {
    __m256 grad_vec = _mm256_set1_ps(output_grad);
    __m256 max_vec = _mm256_set1_ps(max_val);

    int i = 0;
    for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH) {
      __m256 data_vec = _mm256_loadu_ps(&in->data[i]);
      __m256 mask = _mm256_cmp_ps(data_vec, max_vec, _CMP_EQ_OQ);
      __m256 grad_contrib = _mm256_and_ps(grad_vec, mask);
      __m256 in_grad = _mm256_loadu_ps(&in->grad[i]);
      __m256 new_grad = _mm256_add_ps(in_grad, grad_contrib);
      _mm256_storeu_ps(&in->grad[i], new_grad);
    }

    for (; i < in_size; ++i) {
      if (in->data[i] == max_val) {
        in->grad[i] += output_grad;
      }
    }
  } else {
    int *in_strides = in->strides;
    int *in_shape = in->shape;
    int in_ndim = in->ndim;

    for (int linear_idx = 0; linear_idx < in_size; ++linear_idx) {
      int temp_idx = linear_idx;
      int in_offset = 0;

      for (int d = in_ndim - 1; d >= 0; --d) {
        int coord = temp_idx % in_shape[d];
        temp_idx /= in_shape[d];
        in_offset += coord * in_strides[d];
      }

      if (in->data[in_offset] == max_val) {
        in->grad[in_offset] += output_grad;
      }
    }
  }
}


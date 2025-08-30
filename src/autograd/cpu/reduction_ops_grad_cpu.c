#include <immintrin.h>
#include <stdlib.h> // For malloc/free if I decide to use it, but I'm avoiding it.

#include "autograd/autograd.h"
#include "autograd/autograd_utils.h"
#include "utils.h" // Added this for numel, is_contiguous

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
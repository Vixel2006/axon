#include <immintrin.h>

#include "autograd.h"

#define SIMD_WIDTH 8

int get_reduced_dim(int *in_shape, int *out_shape, int in_ndim, int out_ndim) {
  int reduced_dim = -1;
  for (int i = 0; i < in_ndim; ++i) {
    if (out_ndim <= i || in_shape[i] != out_shape[i]) {
      reduced_dim = i;
      break;
    }
  }

  return reduced_dim;
}

int get_num_batches(int *in_shape, int in_ndim, int reduced_dim) {
  int num_batches = 1;
  for (int i = 0; i < in_ndim; ++i) {
    if (i != reduced_dim) num_batches *= in_shape[i];
  }
  return num_batches;
}

void sum_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

  if (reduced_dim == -1) {
    return;
  }

  int reduce_size = in->shape[reduced_dim];

  int num_batches = get_num_batches(in->shape, in->ndim, reduced_dim);

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

void mean_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

  if (reduced_dim == -1) {
    return;
  }

  int reduce_size = in->shape[reduced_dim];

  int num_batches = get_num_batches(in->shape, in->ndim, reduced_dim);

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
      in->grad[base_offset + i] += grad;
    }
  }
}

void max_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];

  int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

  if (reduced_dim == -1) {
    return;
  }

  int reduce_size = in->shape[reduced_dim];

  int num_batches = get_num_batches(in->shape, in->ndim, reduced_dim);

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
        in->grad[base_offset + i] = grad;
      }
    }
  }
}
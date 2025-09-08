#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#include "autograd/autograd.h"

void relu_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  DEBUG_PRINT("relu_grad_op: Computing gradient for ReLU\n");

  Tensor *a = prev[0];

  int size = numel(a->shape, a->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (!is_contiguous(a) || !is_contiguous(out)) {
    if (a->requires_grad) {
      int *a_strides = a->strides;
      int *out_strides = out->strides;
      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          a_offset += coord * a_strides[d];
          out_offset += coord * out_strides[d];
        }
        if (a->data->ptr[a_offset] > 0) {
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset];
        }
      }
    }
  } else {
    int i = 0;
    for (; i + 7 < size; i += 8) {
      __m256 va = _mm256_loadu_ps(a->data->ptr + i);
      __m256 mask = _mm256_cmp_ps(va, _mm256_setzero_ps(), _CMP_GT_OQ);
      __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
      __m256 dmasked = _mm256_and_ps(dout, mask);
      __m256 dprev = _mm256_loadu_ps(a->grad->ptr + i);
      dprev = _mm256_add_ps(dprev, dmasked);
      _mm256_storeu_ps(a->grad->ptr + i, dprev);
    }

    for (; i < size; ++i) {
      if (a->data->ptr[i] > 0) {
        a->grad->ptr[i] += out->grad->ptr[i];
      }
    }
  }
}

void log_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  DEBUG_PRINT("log_grad_op: Computing gradient for natural logarithm\n");

  Tensor *a = prev[0];

  int size = numel(a->shape, a->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (!is_contiguous(a) || !is_contiguous(out)) {
    if (a->requires_grad) {
      int *a_strides = a->strides;
      int *out_strides = out->strides;
      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          a_offset += coord * a_strides[d];
          out_offset += coord * out_strides[d];
        }
        a->grad->ptr[a_offset] +=
            out->grad->ptr[out_offset] / a->data->ptr[a_offset];
      }
    }
  } else {
    int i = 0;
    for (; i + 7 < size; i += 8) {
      __m256 va = _mm256_loadu_ps(a->data->ptr + i);
      __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
      __m256 da = _mm256_loadu_ps(a->grad->ptr + i);

      __m256 inv = _mm256_rcp_ps(va);
      __m256 contrib = _mm256_mul_ps(dout, inv);

      da = _mm256_add_ps(da, contrib);
      _mm256_storeu_ps(a->grad->ptr + i, da);
    }

    for (; i < size; ++i) {
      a->grad->ptr[i] += out->grad->ptr[i] / a->data->ptr[i];
    }
  }
}

void exp_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  DEBUG_PRINT("exp_grad_op: Computing gradient for exponential\n");

  Tensor *a = prev[0];

  int size = numel(a->shape, a->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (!is_contiguous(a) || !is_contiguous(out)) {
    if (a->requires_grad) {
      int *a_strides = a->strides;
      int *out_strides = out->strides;
      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          a_offset += coord * a_strides[d];
          out_offset += coord * out_strides[d];
        }
        a->grad->ptr[a_offset] +=
            out->grad->ptr[out_offset] * out->data->ptr[out_offset];
      }
    }
  } else {
    int i = 0;
    for (; i + 7 < size; i += 8) {
      __m256 vout = _mm256_loadu_ps(out->data->ptr + i);
      __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
      __m256 da = _mm256_loadu_ps(a->grad->ptr + i);

      __m256 contrib = _mm256_mul_ps(dout, vout);

      da = _mm256_add_ps(da, contrib);
      _mm256_storeu_ps(a->grad->ptr + i, da);
    }

    for (; i < size; ++i) {
      a->grad->ptr[i] += out->grad->ptr[i] * out->data->ptr[i];
    }
  }
}

void softmax_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {}

void neg_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  DEBUG_PRINT("neg_grad_op: Computing gradient for negation\n");

  Tensor *a = prev[0];

  int size = numel(a->shape, a->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (!is_contiguous(a) || !is_contiguous(out)) {
    if (a->requires_grad) {
      int *a_strides = a->strides;
      int *out_strides = out->strides;
      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          a_offset += coord * a_strides[d];
          out_offset += coord * out_strides[d];
        }
        a->grad->ptr[a_offset] += -out->grad->ptr[out_offset];
      }
    }
  } else {
    int i = 0;

    __m256 neg_one = _mm256_set1_ps(-1.0f);

    for (; i + 7 < size; i += 8) {
      __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
      __m256 da = _mm256_loadu_ps(a->grad->ptr + i);

      __m256 contrib = _mm256_mul_ps(dout, neg_one);

      da = _mm256_add_ps(da, contrib);

      _mm256_storeu_ps(a->grad->ptr + i, da);
    }

    for (; i < size; ++i) {
      a->grad->ptr[i] += -out->grad->ptr[i];
    }
  }
}

void abs_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  DEBUG_PRINT("abs_grad_op: Computing gradient for absolute value\n");

  Tensor *a = prev[0];
  int size = numel(a->shape, a->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (!is_contiguous(a) || !is_contiguous(out)) {
    if (a->requires_grad) {
      int *a_strides = a->strides;
      int *out_strides = out->strides;
      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
          int coord = idx % shape[d];
          idx /= shape[d];

          a_offset += coord * a_strides[d];
          out_offset += coord * out_strides[d];
        }
        float x = a->data->ptr[a_offset];
        float s = (x > 0) ? 1.0f : (x < 0 ? -1.0f : 0.0f);
        a->grad->ptr[a_offset] += out->grad->ptr[out_offset] * s;
      }
    }
  } else {
    int i = 0;

    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg1 = _mm256_set1_ps(-1.0f);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
      __m256 da = _mm256_loadu_ps(a->grad->ptr + i);

      __m256 mask_pos = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
      __m256 mask_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

      __m256 sign = _mm256_blendv_ps(zero, one, mask_pos);
      sign = _mm256_blendv_ps(sign, neg1, mask_neg);

      __m256 contrib = _mm256_mul_ps(dout, sign);

      da = _mm256_add_ps(da, contrib);

      _mm256_storeu_ps(a->grad->ptr + i, da);
    }

    for (; i < size; ++i) {
      float x = a->data->ptr[i];
      float s = (x > 0) ? 1.0f : (x < 0 ? -1.0f : 0.0f);
      a->grad->ptr[i] += out->grad->ptr[i] * s;
    }
  }
}

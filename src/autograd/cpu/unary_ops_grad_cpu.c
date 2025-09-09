#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#include "autograd/autograd.h"

void relu_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  IDRAK_DEBUG("GRAD ", "relu_grad_op: Computing gradient for ReLU\n");

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
  IDRAK_DEBUG("GRAD ",
      "log_grad_op: Computing gradient for natural logarithm\n");

  // Error checking for null tensors and invalid n_prev
  if (!out || !out->grad || !prev) {
    IDRAK_ERROR("log_grad_op ERROR: Output tensor, output gradient, or previous tensors array is NULL! out=%p, out->grad=%p, prev=%p\n", (void*)out, (void*)out->grad, (void*)prev);
    return;
  }

  if (n_prev != 1) {
    IDRAK_ERROR("log_grad_op ERROR: Invalid number of previous tensors: %d. Expected 1.\n", n_prev);
    return;
  }

  if (!prev[0]) {
    IDRAK_ERROR("log_grad_op ERROR: Previous tensor is NULL! prev[0]=%p\n", (void*)prev[0]);
    return;
  }

  Tensor *a = prev[0];

  if (a->requires_grad && !a->grad) {
    IDRAK_ERROR("log_grad_op ERROR: Tensor 'a' requires grad but its grad is NULL!\n");
    return;
  }

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
        if (a->data->ptr[a_offset] == 0.0f) {
          IDRAK_ERROR("log_grad_op ERROR: Division by zero in gradient at index %d! Input to log was zero.\n", linear);
          // Handle this error, e.g., set gradient to 0 or NaN
          a->grad->ptr[a_offset] +=
              0.0f; // Or NAN
        } else {
          a->grad->ptr[a_offset] +=
              out->grad->ptr[out_offset] / a->data->ptr[a_offset];
        }
      }
    }
  } else {
    int i = 0;
    for (; i + 7 < size; i += 8) {
      __m256 va = _mm256_loadu_ps(a->data->ptr + i);
      __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
      __m256 da = _mm256_loadu_ps(a->grad->ptr + i);

      // Check for zero values in va to avoid division by zero
      __m256 zero = _mm256_setzero_ps();
      __m256 zero_mask = _mm256_cmp_ps(va, zero, _CMP_EQ_OQ);

      // Replace zero values in va with a small epsilon to avoid Inf/NaN from rcp_ps
      __m256 va_safe = _mm256_blendv_ps(va, _mm256_set1_ps(1e-8f), zero_mask);

      __m256 inv = _mm256_rcp_ps(va_safe);
      __m256 contrib = _mm256_mul_ps(dout, inv);

      // If va was originally zero, set contrib to zero for that lane
      contrib = _mm256_blendv_ps(contrib, zero, zero_mask);

      da = _mm256_add_ps(da, contrib);
      _mm256_storeu_ps(a->grad->ptr + i, da);
    }

    for (; i < size; ++i) {
      if (a->data->ptr[i] == 0.0f) {
        IDRAK_ERROR("log_grad_op ERROR: Division by zero in gradient at index %d! Input to log was zero.\n", i);
        a->grad->ptr[i] += 0.0f; // Or NAN
      } else {
        a->grad->ptr[i] += out->grad->ptr[i] / a->data->ptr[i];
      }
    }
  }
}

void exp_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  IDRAK_DEBUG("GRAD ",
      "exp_grad_op: Computing gradient for exponential\n");

  // Error checking for null tensors and invalid n_prev
  if (!out || !out->grad || !prev) {
    IDRAK_ERROR("exp_grad_op ERROR: Output tensor, output gradient, or previous tensors array is NULL! out=%p, out->grad=%p, prev=%p\n", (void*)out, (void*)out->grad, (void*)prev);
    return;
  }

  if (n_prev != 1) {
    IDRAK_ERROR("exp_grad_op ERROR: Invalid number of previous tensors: %d. Expected 1.\n", n_prev);
    return;
  }

  if (!prev[0]) {
    IDRAK_ERROR("exp_grad_op ERROR: Previous tensor is NULL! prev[0]=%p\n", (void*)prev[0]);
    return;
  }

  Tensor *a = prev[0];

  if (a->requires_grad && !a->grad) {
    IDRAK_ERROR("exp_grad_op ERROR: Tensor 'a' requires grad but its grad is NULL!\n");
    return;
  }

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
  IDRAK_DEBUG("GRAD ", "neg_grad_op: Computing gradient for negation\n");

  // Error checking for null tensors and invalid n_prev
  if (!out || !out->grad || !prev) {
    IDRAK_ERROR("neg_grad_op ERROR: Output tensor, output gradient, or previous tensors array is NULL! out=%p, out->grad=%p, prev=%p\n", (void*)out, (void*)out->grad, (void*)prev);
    return;
  }

  if (n_prev != 1) {
    IDRAK_ERROR("neg_grad_op ERROR: Invalid number of previous tensors: %d. Expected 1.\n", n_prev);
    return;
  }

  if (!prev[0]) {
    IDRAK_ERROR("neg_grad_op ERROR: Previous tensor is NULL! prev[0]=%p\n", (void*)prev[0]);
    return;
  }

  Tensor *a = prev[0];

  if (a->requires_grad && !a->grad) {
    IDRAK_ERROR("neg_grad_op ERROR: Tensor 'a' requires grad but its grad is NULL!\n");
    return;
  }

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

  IDRAK_DEBUG("GRAD ",
      "abs_grad_op: Computing gradient for absolute value\n");

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

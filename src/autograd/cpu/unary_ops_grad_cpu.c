#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#include "ops.h"

/**
 * @brief Backward pass for ReLU activation.
 *
 * @param out The output tensor from the forward pass (contains gradient).
 * @param prev Array of input tensors (only one for ReLU).
 * @param n_prev Number of previous tensors (should be 1).
 * @param extras Unused (set to NULL).
 *
 * @effect Accumulates gradient into `prev[0]->grad`, masking out non-positive
 * values.
 */
void relu_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int i = 0;
  int size = numel(a->shape, a->ndim);
  for (; i + 7 < size; i += 8) {
    __m256 va = _mm256_loadu_ps(a->data + i);
    __m256 mask = _mm256_cmp_ps(va, _mm256_setzero_ps(), _CMP_GT_OQ);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 dmasked = _mm256_and_ps(dout, mask);
    __m256 dprev = _mm256_loadu_ps(a->grad + i);
    dprev = _mm256_add_ps(dprev, dmasked);
    _mm256_storeu_ps(a->grad + i, dprev);
  }

  for (; i < size; ++i) {
    if (a->data[i] > 0) {
      a->grad[i] += out->grad[i];
    }
  }
}

/**
 * @brief Backward pass for natural logarithm (log).
 *
 * @param out The output tensor from the forward pass (contains gradient).
 * @param prev Array of input tensors (only one for log).
 * @param n_prev Number of previous tensors (should be 1).
 * @param extras Unused (set to NULL).
 *
 * @effect Accumulates gradient into `prev[0]->grad` using derivative d/dx
 * log(x) = 1/x.
 */
void log_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int i = 0;
  int size = numel(a->shape, a->ndim);
  for (; i + 7 < size; i += 8) {
    __m256 va = _mm256_loadu_ps(a->data + i);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 inv = _mm256_rcp_ps(va);
    __m256 contrib = _mm256_mul_ps(dout, inv);

    da = _mm256_add_ps(da, contrib);
    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    a->grad[i] += out->grad[i] / a->data[i];
  }
}

/**
 * @brief Backward pass for exponential function (exp).
 *
 * @param out The output tensor from the forward pass (contains gradient and
 * data).
 * @param prev Array of input tensors (only one for exp).
 * @param n_prev Number of previous tensors (should be 1).
 * @param extras Unused (set to NULL).
 *
 * @effect Accumulates gradient into `prev[0]->grad` using derivative d/dx
 * exp(x) = exp(x).
 */
void exp_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int i = 0;
  int size = numel(a->shape, a->ndim);
  for (; i + 7 < size; i += 8) {
    __m256 vout = _mm256_loadu_ps(out->data + i);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 contrib = _mm256_mul_ps(dout, vout);

    da = _mm256_add_ps(da, contrib);
    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    a->grad[i] += out->grad[i] * out->data[i];
  }
}

void softmax_grad_op(Tensor *in, Tensor *out) {}

/**
 * @brief Backward pass for negation (-x).
 *
 * @param out The output tensor from the forward pass (contains gradient).
 * @param prev Array of input tensors (only one for neg).
 * @param n_prev Number of previous tensors (should be 1).
 * @param extras Unused (set to NULL).
 *
 * @effect Accumulates gradient into `prev[0]->grad` using derivative d/dx (-x)
 * = -1.
 */
void neg_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];

  int size = numel(a->shape, a->ndim);
  int i = 0;

  __m256 neg_one = _mm256_set1_ps(-1.0f);

  for (; i + 7 < size; i += 8) {
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 contrib = _mm256_mul_ps(dout, neg_one);

    da = _mm256_add_ps(da, contrib);

    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    a->grad[i] += -out->grad[i];
  }
}

/**
 * @brief Backward pass for absolute value (abs).
 *
 * @param out The output tensor from the forward pass (contains gradient).
 * @param prev Array of input tensors (only one for abs).
 * @param n_prev Number of previous tensors (should be 1).
 * @param extras Unused (set to NULL).
 *
 * @effect Accumulates gradient into `prev[0]->grad` using derivative d/dx
 * abs(x) = sign(x).
 */

void abs_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  int size = numel(a->shape, a->ndim);
  int i = 0;

  __m256 zero = _mm256_setzero_ps();
  __m256 one = _mm256_set1_ps(1.0f);
  __m256 neg1 = _mm256_set1_ps(-1.0f);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(a->data + i);
    __m256 dout = _mm256_loadu_ps(out->grad + i);
    __m256 da = _mm256_loadu_ps(a->grad + i);

    __m256 mask_pos = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    __m256 mask_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

    __m256 sign = _mm256_blendv_ps(zero, one, mask_pos);
    sign = _mm256_blendv_ps(sign, neg1, mask_neg);

    __m256 contrib = _mm256_mul_ps(dout, sign);

    da = _mm256_add_ps(da, contrib);

    _mm256_storeu_ps(a->grad + i, da);
  }

  for (; i < size; ++i) {
    float x = a->data[i];
    float s = (x > 0) ? 1.0f : (x < 0 ? -1.0f : 0.0f);
    a->grad[i] += out->grad[i] * s;
  }
}

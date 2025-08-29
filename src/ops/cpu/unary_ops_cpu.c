#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#include "ops/ops.h"

/**
 * @brief Applies the ReLU activation function element-wise.
 *
 * Computes max(0, x) for each element of the input tensor. Uses AVX2
 * intrinsics for vectorized processing, with a scalar fallback for
 * the remainder.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller, same shape as input).
 *
 * @effects Writes element-wise ReLU results into `out->data`.
 * @effects Sets `out->requires_grad = in->requires_grad`.
 * @effects If gradients are not required, frees the input tensor.
 */
void relu_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  __m256 zeros = _mm256_setzero_ps();

  for (; i + 7 < size; i += 8) {
    __m256 vin = _mm256_loadu_ps(in->data + i);
    __m256 vout = _mm256_max_ps(vin, zeros);
    _mm256_storeu_ps(out->data + i, vout);
  }

  for (; i < size; ++i) {
    out->data[i] = in->data[i] > 0.0f ? in->data[i] : 0.0f;
  }

  out->requires_grad = in->requires_grad;
}

/**
 * @brief Computes the natural logarithm element-wise.
 *
 * Applies log(x) to each element of the input tensor. Uses Sleef AVX2
 * vectorized math functions with a scalar fallback.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller, same shape as input).
 *
 * @effects Writes log results into `out->data`.
 * @effects Sets `out->requires_grad = in->requires_grad`.
 * @effects If gradients are not required, frees the input tensor.
 */
void log_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = Sleef_logf8_u10avx2(x);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = logf(in->data[i]);
  }

  out->requires_grad = in->requires_grad;
}

/**
 * @brief Computes the exponential function element-wise.
 *
 * Applies exp(x) to each element of the input tensor. Uses Sleef AVX2
 * intrinsics with a scalar fallback.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller, same shape as input).
 *
 * @effects Writes exponential results into `out->data`.
 * @effects Sets `out->requires_grad = in->requires_grad`.
 * @effects If gradients are not required, frees the input tensor.
 */
void exp_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = Sleef_expf8_u10avx2(x);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = expf(in->data[i]);
  }

  out->requires_grad = in->requires_grad;
}

void softmax_op(Tensor *in, Tensor *out) {}

/**
 * @brief Negates all elements of the input tensor.
 *
 * Computes -x for each element of the input tensor. Uses AVX2 intrinsics
 * for vectorized computation with scalar fallback.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller, same shape as input).
 *
 * @effects Writes negated results into `out->data`.
 * @effects Sets `out->requires_grad = in->requires_grad`.
 * @effects If gradients are not required, frees the input tensor.
 */
void neg_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);

  __m256 zeros = _mm256_setzero_ps();

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = _mm256_sub_ps(zeros, x);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = 0.0f - in->data[i];
  }

  out->requires_grad = in->requires_grad;

  if (!out->requires_grad) {
    free_tensor(in);
  }
}

/**
 * @brief Computes the absolute value element-wise.
 *
 * Applies |x| for each element of the input tensor. Uses AVX2 bitmask
 * trick to clear the sign bit, with scalar fallback.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller, same shape as input).
 *
 * @effects Writes absolute values into `out->data`.
 * @effects Sets `out->requires_grad = in->requires_grad`.
 * @effects If gradients are not required, frees the input tensor.
 */
void abs_op(Tensor *in, Tensor *out) {
  int i = 0;
  int size = numel(in->shape, in->ndim);
  __m256 mask = _mm256_castsi256_ps(
      _mm256_set1_epi32(0x7FFFFFFF));  // mask to remove sign bit

  for (; i + 7 < size; i += 8) {
    __m256 x = _mm256_loadu_ps(in->data + i);
    __m256 y = _mm256_and_ps(x, mask);
    _mm256_storeu_ps(out->data + i, y);
  }

  for (; i < size; ++i) {
    out->data[i] = in->data[i] >= 0 ? in->data[i] : 0.0f - in->data[i];
  }

  out->requires_grad = in->requires_grad;
}

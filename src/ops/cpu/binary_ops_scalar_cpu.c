#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#include "ops/ops.h"

/**
 * @brief Elementwise addition of a tensor and a scalar (SIMD-optimized).
 *
 * Performs elementwise addition of tensor `a` with scalar `b`
 * and stores the result in `out`. Uses AVX2 SIMD (width=8 floats)
 * for bulk computation, with scalar fallback for leftovers.
 *
 * @param a   Input tensor.
 * @param b   Scalar value to add.
 * @param out Output tensor (must be allocated with same shape as `a`).
 *
 * @effects Updates `out->data` with elementwise sum of `a` and `b`.
 * @effects Sets `out->requires_grad` = `a->requires_grad`.
 * @note Assumes contiguous memory layout.
 */
void add_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data[offset_out] = a->data[offset_a] + b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 z = _mm256_add_ps(x, scalar);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] + b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

/**
 * @brief Elementwise subtraction of a scalar from a tensor (SIMD-optimized).
 *
 * Performs elementwise subtraction of scalar `b` from tensor `a`
 * and stores the result in `out`.
 *
 * @param a   Input tensor.
 * @param b   Scalar value to subtract.
 * @param out Output tensor (must be allocated with same shape as `a`).
 *
 * @effects Updates `out->data` with elementwise difference.
 * @effects Sets `out->requires_grad` = `a->requires_grad`.
 * @note Assumes contiguous memory layout.
 */
void sub_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data[offset_out] = a->data[offset_a] - b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 z = _mm256_sub_ps(x, scalar);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] - b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

/**
 * @brief Elementwise reverse subtraction of tensor from a scalar
 * (SIMD-optimized).
 *
 * Computes `a - b[i]` for each element of tensor `b` and stores the result in
 * `out`.
 *
 * @param a   Scalar value.
 * @param b   Input tensor.
 * @param out Output tensor (must be allocated with same shape as `b`).
 *
 * @effects Updates `out->data` with elementwise reverse difference.
 * @effects Sets `out->requires_grad` = `b->requires_grad`.
 * @note Assumes contiguous memory layout.
 */
void rsub_scalar_op(float a, Tensor *b, Tensor *out) {
  int size = numel(b->shape, b->ndim);

  if (!is_contiguous(b) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int b_offset = 0;
      int out_offset = 0;
      int tmp = idx;

      for (int d = b->ndim - 1; d >= 0; --d) {
        int coord = tmp % b->shape[d];
        tmp /= b->shape[d];

        b_offset += coord * b->strides[d];
        out_offset += coord * out->strides[d];
      }

      out->data[out_offset] = a - b->data[b_offset];
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(a);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(b->data + i);
      __m256 z = _mm256_sub_ps(scalar, x);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a - b->data[i];
    }
  }

  out->requires_grad = b->requires_grad ? true : false;
}

/**
 * @brief Elementwise multiplication of a tensor and a scalar (SIMD-optimized).
 *
 * Multiplies each element of tensor `a` by scalar `b` and stores the result in
 * `out`.
 *
 * @param a   Input tensor.
 * @param b   Scalar value to multiply.
 * @param out Output tensor (must be allocated with same shape as `a`).
 *
 * @effects Updates `out->data` with elementwise product.
 * @effects Sets `out->requires_grad` = `a->requires_grad`.
 * @note Assumes contiguous memory layout.
 */
void mul_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data[offset_out] = a->data[offset_a] * b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 z = _mm256_mul_ps(x, scalar);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] * b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

/**
 * @brief Elementwise division of a tensor by a scalar (SIMD-optimized).
 *
 * Divides each element of tensor `a` by scalar `b` and stores the result in
 * `out`.
 *
 * @param a   Input tensor.
 * @param b   Scalar value (divisor).
 * @param out Output tensor (must be allocated with same shape as `a`).
 *
 * @effects Sets `out->requires_grad` = `a->requires_grad`.
 * @note Assumes contiguous memory layout.
 */
void div_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data[offset_out] = a->data[offset_a] / b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 z = _mm256_div_ps(x, scalar);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = a->data[i] / b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

/**
 * @brief Elementwise reverse division of a scalar by a tensor
 * (SIMD-optimized).
 *
 * Computes `a / b[i]` for each element of tensor `b` and stores the result in
 * `out`.
 *
 * @param a   Scalar value (numerator).
 * @param b   Input tensor (denominator).
 * @param out Output tensor (must be allocated with same shape as `b`).
 *
 * @effects Updates `out->data` with elementwise reverse division.
 * @effects Sets `out->requires_grad` = `b->requires_grad`.
 * @note Assumes contiguous memory layout.
 */

void rdiv_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data[offset_out] = b / a->data[offset_a];
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 z = _mm256_div_ps(scalar, x);
      _mm256_storeu_ps(out->data + i, z);
    }

    for (; i < size; ++i) {
      out->data[i] = b / a->data[i];
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void pow_scalar_op(Tensor *a, float b, Tensor *out) {
  int size = numel(a->shape, a->ndim);

  if (!is_contiguous(a) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_a = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = a->ndim - 1; d >= 0; --d) {
        int coord = tmp % a->shape[d];
        tmp /= a->shape[d];
        offset_a += coord * a->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data[offset_out] = pow(a->data[offset_a], b);
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data + i);
      __m256 y = Sleef_powf8_u10avx2(x, scalar);
      _mm256_storeu_ps(out->data + i, y);
    }

    for (; i < size; ++i) {
      out->data[i] = pow(a->data[i], b);
    }
  }
  out->requires_grad = a->requires_grad;
}

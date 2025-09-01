#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <stdio.h>
#include <string.h>

#include "autograd/autograd.h"
#include "ops/ops.h"
#include "utils.h"

/**
 * @brief Helper function for matrix multiplication of raw float arrays.
 *
 * Computes C = A * B, where A is M x K, B is K x N, and C is M x N.
 * Assumes row-major order.
 *
 * @param A Pointer to the data of matrix A.
 * @param B Pointer to the data of matrix B.
 * @param C Pointer to the data of matrix C (output).
 * @param M Number of rows in A and C.
 * @param K Number of columns in A and rows in B.
 * @param N Number of columns in B and C.
 */
static void _matmul_float_arrays(const float *A, const float *B, float *C,
                                 int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < K; ++l) {
        sum += A[i * K + l] * B[l * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

/**
 * @brief Backward pass for addition operation.
 *
 * Accumulates gradients into the inputs of an addition operation.
 * Handles both tensor + tensor and tensor + scalar cases.
 *
 * @param out      Output tensor whose gradient is being propagated.
 * @param prev     Array of input tensors (1 or 2 depending on the case).
 * @param n_prev   Number of input tensors (1 for scalar add, 2 for tensor add).
 * @param extras   Pointer to scalar value if scalar addition was used,
 * otherwise NULL.
 *
 * @effects Modifies `prev[i]->grad` in-place by adding contributions from
 * `out->grad`.
 */
void add_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
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
          a->grad[a_offset] += out->grad[out_offset];
        }
      }

      if (b->requires_grad) {
        int *b_strides = b->strides;
        int *out_strides = out->strides;
        for (int linear = 0; linear < size; ++linear) {
          int idx = linear;
          int b_offset = 0, out_offset = 0;

          for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];

            b_offset += coord * b_strides[d];
            out_offset += coord * out_strides[d];
          }
          b->grad[b_offset] += out->grad[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 db = _mm256_add_ps(b_grad, dout);
          _mm256_storeu_ps(b->grad + i, db);
        }

        for (; i < size; ++i) {
          b->grad[i] += out->grad[i];
        }
      }
    }
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

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
          a->grad[a_offset] += out->grad[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i];
        }
      }
    }
  }
}

/**
 * @brief Backward pass for subtraction operation.
 *
 * Accumulates gradients into the inputs of a subtraction operation.
 * Handles both tensor - tensor and tensor - scalar cases.
 *
 * @param out      Output tensor whose gradient is being propagated.
 * @param prev     Array of input tensors (1 or 2 depending on the case).
 * @param n_prev   Number of input tensors (1 for scalar sub, 2 for tensor sub).
 * @param extras   Pointer to scalar value if scalar subtraction was used,
 * otherwise NULL.
 *
 * @effects Modifies `prev[i]->grad` in-place by adding or subtracting
 * contributions from `out->grad`.
 */
void sub_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
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
          a->grad[a_offset] += out->grad[out_offset];
        }
      }

      if (b->requires_grad) {
        int *b_strides = b->strides;
        int *out_strides = out->strides;
        for (int linear = 0; linear < size; ++linear) {
          int idx = linear;
          int b_offset = 0, out_offset = 0;

          for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];

            b_offset += coord * b_strides[d];
            out_offset += coord * out_strides[d];
          }
          b->grad[b_offset] -= out->grad[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 db = _mm256_sub_ps(b_grad, dout);
          _mm256_storeu_ps(b->grad + i, db);
        }

        for (; i < size; ++i) {
          b->grad[i] -= out->grad[i];
        }
      }
    }
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

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
          a->grad[a_offset] += out->grad[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i];
        }
      }
    }
  }
}

/**
 * @brief Backward pass for reverse subtraction (scalar - tensor).
 *
 * Computes gradients when the forward op was `b - a` (scalar minus tensor).
 *
 * @param out      Output tensor whose gradient is being propagated.
 * @param prev     Array containing the single tensor input.
 * @param n_prev   Should always be 1 for rsub.
 * @param extras   Pointer to scalar value `b` used in forward pass.
 *
 * @effects Subtracts `out->grad` from `a->grad`.
 */
void rsub_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  float b = *((float *)extras);

  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (a->requires_grad) {
    if (!is_contiguous(a) || !is_contiguous(out)) {
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
        a->grad[a_offset] -= out->grad[out_offset];
      }
    } else {
      int i = 0;
      for (; i + 7 < size; i += 8) {
        __m256 a_grad = _mm256_loadu_ps(a->grad + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 da = _mm256_sub_ps(a_grad, dout);
        _mm256_storeu_ps(a->grad + i, da);
      }

      for (; i < size; ++i) {
        a->grad[i] -= out->grad[i];
      }
    }
  }
}

/**
 * @brief Backward pass for multiplication operation.
 *
 * Accumulates gradients into the inputs of a multiplication operation.
 * Handles both tensor * tensor and tensor * scalar cases.
 *
 * @param out      Output tensor whose gradient is being propagated.
 * @param prev     Array of input tensors (1 or 2 depending on the case).
 * @param n_prev   Number of input tensors (1 for scalar mul, 2 for tensor mul).
 * @param extras   Pointer to scalar value if scalar multiplication was used,
 * otherwise NULL.
 *
 * @effects Updates gradients of inputs using the product rule.
 */
void mul_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
      if (a->requires_grad) {
        int *a_strides = a->strides;
        int *b_strides = b->strides;
        int *out_strides = out->strides;
        for (int linear = 0; linear < size; ++linear) {
          int idx = linear;
          int a_offset = 0, b_offset = 0, out_offset = 0;

          for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];

            a_offset += coord * a_strides[d];
            b_offset += coord * b_strides[d];
            out_offset += coord * out_strides[d];
          }
          a->grad[a_offset] += out->grad[out_offset] * b->data[b_offset];
        }
      }

      if (b->requires_grad) {
        int *a_strides = a->strides;
        int *b_strides = b->strides;
        int *out_strides = out->strides;
        for (int linear = 0; linear < size; ++linear) {
          int idx = linear;
          int a_offset = 0, b_offset = 0, out_offset = 0;

          for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];

            a_offset += coord * a_strides[d];
            b_offset += coord * b_strides[d];
            out_offset += coord * out_strides[d];
          }
          b->grad[b_offset] += out->grad[out_offset] * a->data[a_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 b_data = _mm256_loadu_ps(b->data + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_fmadd_ps(b_data, dout, a_grad);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i] * b->data[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad + i);
          __m256 a_data = _mm256_loadu_ps(a->data + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 db = _mm256_fmadd_ps(a_data, dout, b_grad);
          _mm256_storeu_ps(b->grad + i, db);
        }

        for (; i < size; ++i) {
          b->grad[i] += out->grad[i] * a->data[i];
        }
      }
    }
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

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
          a->grad[a_offset] += out->grad[out_offset] * b;
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_fmadd_ps(scalar, dout, a_grad);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i] * b;
        }
      }
    }
  }
}

/**
 * @brief Backward pass for division operation.
 *
 * Accumulates gradients into the inputs of a division operation.
 * Handles both tensor / tensor and tensor / scalar cases.
 *
 * @param out      Output tensor whose gradient is being propagated.
 * @param prev     Array of input tensors (1 or 2 depending on the case).
 * @param n_prev   Number of input tensors (1 for scalar div, 2 for tensor div).
 * @param extras   Pointer to scalar value if scalar division was used,
 * otherwise NULL.
 *
 * @effects Updates gradients of inputs according to the quotient rule.
 */
void div_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
      if (a->requires_grad) {
        int *a_strides = a->strides;
        int *b_strides = b->strides;
        int *out_strides = out->strides;
        for (int linear = 0; linear < size; ++linear) {
          int idx = linear;
          int a_offset = 0, b_offset = 0, out_offset = 0;

          for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];

            a_offset += coord * a_strides[d];
            b_offset += coord * b_strides[d];
            out_offset += coord * out_strides[d];
          }
          a->grad[a_offset] += out->grad[out_offset] / b->data[b_offset];
        }
      }

      if (b->requires_grad) {
        int *a_strides = a->strides;
        int *b_strides = b->strides;
        int *out_strides = out->strides;
        for (int linear = 0; linear < size; ++linear) {
          int idx = linear;
          int a_offset = 0, b_offset = 0, out_offset = 0;

          for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];

            a_offset += coord * a_strides[d];
            b_offset += coord * b_strides[d];
            out_offset += coord * out_strides[d];
          }
          b->grad[b_offset] -= out->grad[out_offset] * a->data[a_offset] /
                               (b->data[b_offset] * b->data[b_offset]);
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 b_data = _mm256_loadu_ps(b->data + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_fmadd_ps(_mm256_div_ps(dout, b_data),
                                      _mm256_set1_ps(1.0f), a_grad);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i] / b->data[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad + i);
          __m256 a_data = _mm256_loadu_ps(a->data + i);
          __m256 b_data = _mm256_loadu_ps(b->data + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 b_squared = _mm256_mul_ps(b_data, b_data);
          __m256 db =
              _mm256_fnmadd_ps(_mm256_div_ps(a_data, b_squared), dout, b_grad);
          _mm256_storeu_ps(b->grad + i, db);
        }

        for (; i < size; ++i) {
          b->grad[i] -= out->grad[i] * a->data[i] / (b->data[i] * b->data[i]);
        }
      }
    }
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

    if (!is_contiguous(a) || !is_contiguous(out)) {
      if (a->requires_grad) {
        int *a_strides = a->strides;
        int *out_strides = out->strides;
        for (int linear = 0; linear = size; ++linear) {
          int idx = linear;
          int a_offset = 0, out_offset = 0;

          for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];

            a_offset += coord * a_strides[d];
            out_offset += coord * out_strides[d];
          }
          a->grad[a_offset] += out->grad[out_offset] / b;
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        __m256 inv_b = _mm256_set1_ps(1.0f / b);
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad + i);
          __m256 dout = _mm256_loadu_ps(out->grad + i);
          __m256 da = _mm256_fmadd_ps(dout, inv_b, a_grad);
          _mm256_storeu_ps(a->grad + i, da);
        }

        for (; i < size; ++i) {
          a->grad[i] += out->grad[i] / b;
        }
      }
    }
  }
}

/**
 * @brief Backward pass for reverse division (scalar / tensor).
 *
 * Computes gradients when the forward op was `b / a` (scalar divided by
 * tensor).
 *
 * @param out      Output tensor whose gradient is being propagated.
 * @param prev     Array containing the single tensor input.
 * @param n_prev   Should always be 1 for rdiv.
 * @param extras   Pointer to scalar value `b` used in forward pass.
 *
 * @effects Updates `a->grad` using `-b / (a^2)` multiplied by `out->grad`.
 */
void rdiv_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  float b = *((float *)extras);

  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  if (a->requires_grad) {
    if (!is_contiguous(a) || !is_contiguous(out)) {
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
        a->grad[a_offset] += out->grad[out_offset] * (-b) /
                             (a->data[a_offset] * a->data[a_offset]);
      }
    } else {
      int i = 0;
      __m256 neg_b = _mm256_set1_ps(-b);
      for (; i + 7 < size; i += 8) {
        __m256 a_grad = _mm256_loadu_ps(a->grad + i);
        __m256 a_data = _mm256_loadu_ps(a->data + i);
        __m256 dout = _mm256_loadu_ps(out->grad + i);
        __m256 a_squared = _mm256_mul_ps(a_data, a_data);
        __m256 da =
            _mm256_fmadd_ps(_mm256_div_ps(neg_b, a_squared), dout, a_grad);
        _mm256_storeu_ps(a->grad + i, da);
      }

      for (; i < size; ++i) {
        a->grad[i] += out->grad[i] * (-b) / (a->data[i] * a->data[i]);
      }
    }
  }
}

void matmul_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  Tensor *b = prev[1];

  int N = a->shape[a->ndim - 2];
  int K = a->shape[a->ndim - 1];
  int M = b->shape[b->ndim - 1];

  int a_strides = a->strides[a->ndim - 2];
  int a_k_strides = a->strides[a->ndim - 1];
  int b_strides = b->strides[b->ndim - 2];
  int b_m_strides = b->strides[b->ndim - 1];
  int out_strides = out->strides[out->ndim - 2];
  int out_m_strides = out->strides[out->ndim - 1];

  if (a->requires_grad) {
    int batch_nums = get_num_batches(a->shape, a->ndim);

    for (int batch_idx = 0; batch_idx < batch_nums; ++batch_idx) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
          float sum = 0.0f;
          for (int m = 0; m < M; ++m) {
            sum += out->grad[batch_idx * out_strides * M + i * out_strides +
                             m * out_m_strides] *
                   b->data[batch_idx * b_strides * M + j * b_strides +
                           m * b_m_strides];
          }
          a->grad[batch_idx * a_strides * K + i * a_strides +
                  j * a_k_strides] += sum;
        }
      }
    }
  }

  if (b->requires_grad) {
    int batch_nums = get_num_batches(b->shape, b->ndim);

    for (int batch_idx = 0; batch_idx < batch_nums; ++batch_idx) {
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
          float sum = 0.0f;
          for (int n = 0; n < N; ++n) {
            sum += a->data[batch_idx * a_strides * K + n * a_strides +
                           i * a_k_strides] *
                   out->grad[batch_idx * out_strides * M + n * out_strides +
                             j * out_m_strides];
          }
          b->grad[batch_idx * b_strides * M + i * b_strides +
                  j * b_m_strides] += sum;
        }
      }
    }
  }
}

typedef struct {
  int padding;
  int H_in;
  int W_in;
  int Kh;
  int Kw;
  int Sh;
  int Sw;
  int Hout;
  int Wout;
} BackwardConvExtras;

void conv2d_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *in = prev[0];
  Tensor *kernel = prev[1];

  BackwardConvExtras *conv_extras = (BackwardConvExtras *)extras;

  int N = in->shape[0];    // Batch size
  int Cin = in->shape[1];  // Input channels
  int Hin = conv_extras->H_in;
  int Win = conv_extras->W_in;
  int Cout = out->shape[1];
  int Kh = conv_extras->Kh;
  int Kw = conv_extras->Kw;
  int Sh = conv_extras->Sh;
  int Sw = conv_extras->Sw;
  int Hout = conv_extras->Hout;
  int Wout = conv_extras->Wout;
  int padding = conv_extras->padding;

  // Calculate dimensions for im2row and matrix multiplications
  int im_rows_M = N * Hout * Wout;
  int im_rows_K = Cin * Kh * Kw;
  int d_out_N = Cout;

  // Temporary buffer for im2row output
  float *im_rows = (float *)malloc(im_rows_M * im_rows_K * sizeof(float));
  if (im_rows == NULL) {
    fprintf(stderr, "Memory allocation failed for im_rows\n");
    return;
  }

  // Reshape input 'in' to im_rows
  im2row(in->data, im_rows, N, Cin, Hin, Win, Kh, Kw, Sh, Sw, Hout, Wout,
         padding);

  // Gradient with respect to kernel (d_kernel)
  if (kernel->requires_grad) {
    // d_out_reshaped: (N * Hout * Wout) x Cout
    float *d_out_reshaped =
        (float *)malloc(im_rows_M * d_out_N * sizeof(float));
    if (d_out_reshaped == NULL) {
      fprintf(stderr, "Memory allocation failed for d_out_reshaped\n");
      free(im_rows);
      return;
    }
    // Copy out->grad to d_out_reshaped, effectively flattening the spatial
    // dimensions out->grad shape: (N, Cout, Hout, Wout) d_out_reshaped shape:
    // (N * Hout * Wout, Cout)
    for (int n = 0; n < N; ++n) {
      for (int co = 0; co < Cout; ++co) {
        for (int oh = 0; oh < Hout; ++oh) {
          for (int ow = 0; ow < Wout; ++ow) {
            int out_grad_idx =
                n * Cout * Hout * Wout + co * Hout * Wout + oh * Wout + ow;
            int d_out_reshaped_idx =
                (n * Hout * Wout + oh * Wout + ow) * Cout + co;
            d_out_reshaped[d_out_reshaped_idx] = out->grad[out_grad_idx];
          }
        }
      }
    }

    // Transpose im_rows for multiplication: (Cin * Kh * Kw) x (N * Hout * Wout)
    float *im_rows_T = (float *)malloc(im_rows_K * im_rows_M * sizeof(float));
    if (im_rows_T == NULL) {
      fprintf(stderr, "Memory allocation failed for im_rows_T\n");
      free(im_rows);
      free(d_out_reshaped);
      return;
    }
    for (int i = 0; i < im_rows_M; ++i) {
      for (int j = 0; j < im_rows_K; ++j) {
        im_rows_T[j * im_rows_M + i] = im_rows[i * im_rows_K + j];
      }
    }

    // Result of multiplication: (Cin * Kh * Kw) x Cout
    float *d_kernel_flat = (float *)malloc(im_rows_K * d_out_N * sizeof(float));
    if (d_kernel_flat == NULL) {
      fprintf(stderr, "Memory allocation failed for d_kernel_flat\n");
      free(im_rows);
      free(d_out_reshaped);
      free(im_rows_T);
      return;
    }

    _matmul_float_arrays(im_rows_T, d_out_reshaped, d_kernel_flat, im_rows_K,
                         im_rows_M, d_out_N);

    // Accumulate to kernel->grad
    // kernel->grad shape: (Cout, Cin, Kh, Kw)
    // d_kernel_flat shape: (Cin * Kh * Kw, Cout)
    for (int co = 0; co < Cout; ++co) {
      for (int cin_kh_kw = 0; cin_kh_kw < Cin * Kh * Kw; ++cin_kh_kw) {
        int kernel_grad_idx = co * Cin * Kh * Kw + cin_kh_kw;
        int d_kernel_flat_idx = cin_kh_kw * Cout + co;
        kernel->grad[kernel_grad_idx] += d_kernel_flat[d_kernel_flat_idx];
      }
    }

    free(d_out_reshaped);
    free(im_rows_T);
    free(d_kernel_flat);
  }

  // Gradient with respect to input (d_in)
  if (in->requires_grad) {
    // Reshape kernel for multiplication: (Cout) x (Cin * Kh * Kw)
    // This is already the kernel's shape (Cout, Cin, Kh, Kw) flattened
    // kernel->data shape: (Cout, Cin, Kh, Kw)
    // d_out_reshaped: (N * Hout * Wout) x Cout (from above, or recompute if not
    // done)
    float *d_out_reshaped_for_in = NULL;
    if (!kernel->requires_grad) {  // Only recompute if not already done for
                                   // kernel grad
      d_out_reshaped_for_in =
          (float *)malloc(im_rows_M * d_out_N * sizeof(float));
      if (d_out_reshaped_for_in == NULL) {
        fprintf(stderr, "Memory allocation failed for d_out_reshaped_for_in\n");
        free(im_rows);
        return;
      }
      for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
          for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
              int out_grad_idx =
                  n * Cout * Hout * Wout + co * Hout * Wout + oh * Wout + ow;
              int d_out_reshaped_idx =
                  (n * Hout * Wout + oh * Wout + ow) * Cout + co;
              d_out_reshaped_for_in[d_out_reshaped_idx] =
                  out->grad[out_grad_idx];
            }
          }
        }
      }
    } else {
      // If kernel->requires_grad was true, d_out_reshaped is already computed
      // and valid We need to ensure it's not freed prematurely if we reuse it.
      // For simplicity and to avoid complex ownership, recompute or pass it.
      // For now, let's assume it's recomputed if needed.
      // A more optimized solution would pass it around or use a shared pointer.
      d_out_reshaped_for_in =
          (float *)malloc(im_rows_M * d_out_N * sizeof(float));
      if (d_out_reshaped_for_in == NULL) {
        fprintf(stderr, "Memory allocation failed for d_out_reshaped_for_in\n");
        free(im_rows);
        return;
      }
      for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
          for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
              int out_grad_idx =
                  n * Cout * Hout * Wout + co * Hout * Wout + oh * Wout + ow;
              int d_out_reshaped_idx =
                  (n * Hout * Wout + oh * Wout + ow) * Cout + co;
              d_out_reshaped_for_in[d_out_reshaped_idx] =
                  out->grad[out_grad_idx];
            }
          }
        }
      }
    }

    // Transpose kernel for multiplication: (Cin * Kh * Kw) x Cout
    // This is the same shape as d_kernel_flat, but with kernel data
    float *kernel_T = (float *)malloc(im_rows_K * Cout * sizeof(float));
    if (kernel_T == NULL) {
      fprintf(stderr, "Memory allocation failed for kernel_T\n");
      free(im_rows);
      if (!kernel->requires_grad) free(d_out_reshaped_for_in);
      return;
    }
    // kernel->data shape: (Cout, Cin, Kh, Kw)
    // kernel_T shape: (Cin * Kh * Kw, Cout)
    for (int co = 0; co < Cout; ++co) {
      for (int cin_kh_kw = 0; cin_kh_kw < Cin * Kh * Kw; ++cin_kh_kw) {
        int kernel_data_idx = co * Cin * Kh * Kw + cin_kh_kw;
        int kernel_T_idx = cin_kh_kw * Cout + co;
        kernel_T[kernel_T_idx] = kernel->data[kernel_data_idx];
      }
    }

    // Result of multiplication: (N * Hout * Wout) x (Cin * Kh * Kw)
    float *d_in_im_rows =
        (float *)malloc(im_rows_M * im_rows_K * sizeof(float));
    if (d_in_im_rows == NULL) {
      fprintf(stderr, "Memory allocation failed for d_in_im_rows\n");
      free(im_rows);
      if (!kernel->requires_grad) free(d_out_reshaped_for_in);
      free(kernel_T);
      return;
    }

    _matmul_float_arrays(d_out_reshaped_for_in, kernel_T, d_in_im_rows,
                         im_rows_M, Cout, im_rows_K);

    // Allocate temporary buffer for col2im output
    int in_grad_size = N * Cin * Hin * Win;
    float *d_in_temp = (float *)malloc(in_grad_size * sizeof(float));
    if (d_in_temp == NULL) {
      fprintf(stderr, "Memory allocation failed for d_in_temp\n");
      free(im_rows);
      if (!kernel->requires_grad) free(d_out_reshaped_for_in);
      free(kernel_T);
      free(d_in_im_rows);
      return;
    }

    // Use col2im to transform d_in_im_rows to d_in_temp
    col2im(d_in_im_rows, d_in_temp, N, Cin, Hin, Win, Kh, Kw, Sh, Sw, Hout,
           Wout, padding);

    // Accumulate d_in_temp to in->grad
    for (int i = 0; i < in_grad_size; ++i) {
      in->grad[i] += d_in_temp[i];
    }

    free(d_in_temp);
    if (!kernel->requires_grad) free(d_out_reshaped_for_in);
    free(kernel_T);
    free(d_in_im_rows);
  }

  free(im_rows);
}

#include <immintrin.h>
#include <math.h>
#include <sleef.h>

#include "autograd/autograd.h"
#include "utils.h"
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

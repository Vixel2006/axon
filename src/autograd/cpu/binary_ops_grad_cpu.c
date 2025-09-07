#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <stdio.h>
#include <string.h>

#include "autograd/autograd.h"
#include "ops/ops.h"
#include "utils.h"

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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset];
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
          b->grad->ptr[b_offset] += out->grad->ptr[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 db = _mm256_add_ps(b_grad, dout);
          _mm256_storeu_ps(b->grad->ptr + i, db);
        }

        for (; i < size; ++i) {
          b->grad->ptr[i] += out->grad->ptr[i];
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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i];
        }
      }
    }
  }
}

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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset];
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
          b->grad->ptr[b_offset] -= out->grad->ptr[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 db = _mm256_sub_ps(b_grad, dout);
          _mm256_storeu_ps(b->grad->ptr + i, db);
        }

        for (; i < size; ++i) {
          b->grad->ptr[i] -= out->grad->ptr[i];
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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_add_ps(a_grad, dout);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i];
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
 * @effects Subtracts `out->grad->ptr` from `a->grad->ptr`.
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
        a->grad->ptr[a_offset] -= out->grad->ptr[out_offset];
      }
    } else {
      int i = 0;
      for (; i + 7 < size; i += 8) {
        __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
        __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
        __m256 da = _mm256_sub_ps(a_grad, dout);
        _mm256_storeu_ps(a->grad->ptr + i, da);
      }

      for (; i < size; ++i) {
        a->grad->ptr[i] -= out->grad->ptr[i];
      }
    }
  }
}

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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset] * b->data->ptr[b_offset];
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
          b->grad->ptr[b_offset] += out->grad->ptr[out_offset] * a->data->ptr[a_offset];
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 b_data = _mm256_loadu_ps(b->data->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_fmadd_ps(b_data, dout, a_grad);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i] * b->data->ptr[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad->ptr + i);
          __m256 a_data = _mm256_loadu_ps(a->data->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 db = _mm256_fmadd_ps(a_data, dout, b_grad);
          _mm256_storeu_ps(b->grad->ptr + i, db);
        }

        for (; i < size; ++i) {
          b->grad->ptr[i] += out->grad->ptr[i] * a->data->ptr[i];
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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset] * b;
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_fmadd_ps(scalar, dout, a_grad);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i] * b;
        }
      }
    }
  }
}

void pow_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  Tensor *a = prev[0];
  float b = *((float *)extras);

  if (!a->requires_grad)
    return;

  if (!is_contiguous(a) || !is_contiguous(out)) {
    // Non-contiguous fallback
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
      float grad_val = 0.0f;

      // numerical stability check
      if (!(x == 0.0f && (b - 1.0f) < 0.0f)) {
        grad_val = b * powf(x, b - 1.0f);
      }
      a->grad->ptr[a_offset] += out->grad->ptr[out_offset] * grad_val;
    }
  } else {
    int i = 0;
    __m256 scalar_b = _mm256_set1_ps(b);
    float c = b - 1.0f;
    __m256 scalar_bm1 = _mm256_set1_ps(c);
    __m256 zero = _mm256_setzero_ps();

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
      __m256 agrad = _mm256_loadu_ps(a->grad->ptr + i);

      __m256 x_pow = Sleef_powf8_u10avx2(x, scalar_bm1);
      __m256 coeff = _mm256_mul_ps(scalar_b, x_pow);

      // mask problematic case: x==0 && (b-1)<0
      __m256 zero_mask = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
      __m256 neg_exp_mask = _mm256_cmp_ps(scalar_bm1, zero, _CMP_LT_OQ);
      __m256 problem_mask = _mm256_and_ps(zero_mask, neg_exp_mask);
      coeff = _mm256_blendv_ps(coeff, zero, problem_mask);

      __m256 da = _mm256_fmadd_ps(dout, coeff, agrad);
      _mm256_storeu_ps(a->grad->ptr + i, da);
    }

    for (; i < size; ++i) {
      float x = a->data->ptr[i];
      float grad_val = 0.0f;

      if (!(x == 0.0f && (b - 1.0f) < 0.0f)) {
        grad_val = b * powf(x, c);
      }
      a->grad->ptr[i] += out->grad->ptr[i] * grad_val;
    }
  }
}

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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset] / b->data->ptr[b_offset];
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
          b->grad->ptr[b_offset] -= out->grad->ptr[out_offset] * a->data->ptr[a_offset] /
                               (b->data->ptr[b_offset] * b->data->ptr[b_offset]);
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 b_data = _mm256_loadu_ps(b->data->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_fmadd_ps(_mm256_div_ps(dout, b_data),
                                      _mm256_set1_ps(1.0f), a_grad);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i] / b->data->ptr[i];
        }
      }

      if (b->requires_grad) {
        int i = 0;
        for (; i + 7 < size; i += 8) {
          __m256 b_grad = _mm256_loadu_ps(b->grad->ptr + i);
          __m256 a_data = _mm256_loadu_ps(a->data->ptr + i);
          __m256 b_data = _mm256_loadu_ps(b->data->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 b_squared = _mm256_mul_ps(b_data, b_data);
          __m256 db =
              _mm256_fnmadd_ps(_mm256_div_ps(a_data, b_squared), dout, b_grad);
          _mm256_storeu_ps(b->grad->ptr + i, db);
        }

        for (; i < size; ++i) {
          b->grad->ptr[i] -= out->grad->ptr[i] * a->data->ptr[i] / (b->data->ptr[i] * b->data->ptr[i]);
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
          a->grad->ptr[a_offset] += out->grad->ptr[out_offset] / b;
        }
      }
    } else {
      if (a->requires_grad) {
        int i = 0;
        __m256 inv_b = _mm256_set1_ps(1.0f / b);
        for (; i + 7 < size; i += 8) {
          __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
          __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
          __m256 da = _mm256_fmadd_ps(dout, inv_b, a_grad);
          _mm256_storeu_ps(a->grad->ptr + i, da);
        }

        for (; i < size; ++i) {
          a->grad->ptr[i] += out->grad->ptr[i] / b;
        }
      }
    }
  }
}

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
        a->grad->ptr[a_offset] += out->grad->ptr[out_offset] * (-b) /
                             (a->data->ptr[a_offset] * a->data->ptr[a_offset]);
      }
    } else {
      int i = 0;
      __m256 neg_b = _mm256_set1_ps(-b);
      for (; i + 7 < size; i += 8) {
        __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
        __m256 a_data = _mm256_loadu_ps(a->data->ptr + i);
        __m256 dout = _mm256_loadu_ps(out->grad->ptr + i);
        __m256 a_squared = _mm256_mul_ps(a_data, a_data);
        __m256 da =
            _mm256_fmadd_ps(_mm256_div_ps(neg_b, a_squared), dout, a_grad);
        _mm256_storeu_ps(a->grad->ptr + i, da);
      }

      for (; i < size; ++i) {
        a->grad->ptr[i] += out->grad->ptr[i] * (-b) / (a->data->ptr[i] * a->data->ptr[i]);
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
            sum += out->grad->ptr[batch_idx * out_strides * M + i * out_strides +
                             m * out_m_strides] *
                   b->data->ptr[batch_idx * b_strides * M + j * b_strides +
                           m * b_m_strides];
          }
          a->grad->ptr[batch_idx * a_strides * K + i * a_strides +
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
            sum += a->data->ptr[batch_idx * a_strides * K + n * a_strides +
                           i * a_k_strides] *
                   out->grad->ptr[batch_idx * out_strides * M + n * out_strides +
                             j * out_m_strides];
          }
          b->grad->ptr[batch_idx * b_strides * M + i * b_strides +
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

  int N = in->shape[0];
  int Cin = in->shape[1];
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

  const int TILE_H = 16;
  const int TILE_W = 16;

  if (kernel->requires_grad) {
    for (int n = 0; n < N; ++n) {
      for (int oh_start = 0; oh_start < Hout; oh_start += TILE_H) {
        int oh_end = (oh_start + TILE_H > Hout) ? Hout : oh_start + TILE_H;

        for (int ow_start = 0; ow_start < Wout; ow_start += TILE_W) {
          int ow_end = (ow_start + TILE_W > Wout) ? Wout : ow_start + TILE_W;

          for (int kh = 0; kh < Kh; ++kh) {
            for (int kw = 0; kw < Kw; ++kw) {
              for (int oh = oh_start; oh < oh_end; ++oh) {
                for (int ow = ow_start; ow < ow_end; ++ow) {
                  int ih = oh * Sh - padding + kh;
                  int iw = ow * Sw - padding + kw;

                  if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                    for (int cout = 0; cout < Cout; ++cout) {
                      float out_grad_val =
                          out->grad->ptr[n * Cout * Hout * Wout +
                                    cout * Hout * Wout + oh * Wout + ow];

                      for (int cin = 0; cin < Cin; ++cin) {
                        int in_idx = n * Cin * Hin * Win + cin * Hin * Win +
                                     ih * Win + iw;
                        int kernel_grad_idx =
                            cout * Cin * Kh * Kw + cin * Kh * Kw + kh * Kw + kw;

                        kernel->grad->ptr[kernel_grad_idx] +=
                            in->data->ptr[in_idx] * out_grad_val;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (in->requires_grad) {
    for (int n = 0; n < N; ++n) {
      for (int cout = 0; cout < Cout; ++cout) {
        for (int kh = 0; kh < Kh; ++kh) {
          for (int kw = 0; kw < Kw; ++kw) {
            for (int oh_start = 0; oh_start < Hout; oh_start += TILE_H) {
              int oh_end =
                  (oh_start + TILE_H > Hout) ? Hout : oh_start + TILE_H;

              for (int ow_start = 0; ow_start < Wout; ow_start += TILE_W) {
                int ow_end =
                    (ow_start + TILE_W > Wout) ? Wout : ow_start + TILE_W;

                for (int oh = oh_start; oh < oh_end; ++oh) {
                  for (int ow = ow_start; ow < ow_end; ++ow) {
                    int ih = oh * Sh - padding + kh;
                    int iw = ow * Sw - padding + kw;

                    if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                      float out_grad_val =
                          out->grad->ptr[n * Cout * Hout * Wout +
                                    cout * Hout * Wout + oh * Wout + ow];

                      for (int cin = 0; cin < Cin; ++cin) {
                        int kernel_idx =
                            cout * Cin * Kh * Kw + cin * Kh * Kw + kh * Kw + kw;
                        int in_grad_idx = n * Cin * Hin * Win +
                                          cin * Hin * Win + ih * Win + iw;

                        in->grad->ptr[in_grad_idx] +=
                            kernel->data->ptr[kernel_idx] * out_grad_val;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void dot_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  Tensor *b = prev[1];

  int size = numel(a->shape, a->ndim);
  float dout = out->grad->ptr[0];

  if (!is_contiguous(a) || !is_contiguous(b)) {
    if (a->requires_grad) {
      int *a_strides = a->strides;
      int *b_strides = b->strides;
      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, b_offset = 0;

        for (int d = a->ndim - 1; d >= 0; --d) {
          int coord = idx % a->shape[d];
          idx /= a->shape[d];

          a_offset += coord * a_strides[d];
          b_offset += coord * b_strides[d];
        }
        a->grad->ptr[a_offset] += dout * b->data->ptr[b_offset];
      }
    }

    if (b->requires_grad) {
      int *a_strides = a->strides;
      int *b_strides = b->strides;
      for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, b_offset = 0;

        for (int d = a->ndim - 1; d >= 0; --d) {
          int coord = idx % a->shape[d];
          idx /= a->shape[d];

          a_offset += coord * a_strides[d];
          b_offset += coord * b_strides[d];
        }
        b->grad->ptr[b_offset] += dout * a->data->ptr[a_offset];
      }
    }
  } else {
    if (a->requires_grad) {
      int i = 0;
      __m256 dout_vec = _mm256_set1_ps(dout);
      for (; i + 7 < size; i += 8) {
        __m256 a_grad = _mm256_loadu_ps(a->grad->ptr + i);
        __m256 b_data = _mm256_loadu_ps(b->data->ptr + i);
        __m256 da = _mm256_fmadd_ps(dout_vec, b_data, a_grad);
        _mm256_storeu_ps(a->grad->ptr + i, da);
      }

      for (; i < size; ++i) {
        a->grad->ptr[i] += dout * b->data->ptr[i];
      }
    }

    if (b->requires_grad) {
      int i = 0;
      __m256 dout_vec = _mm256_set1_ps(dout);
      for (; i + 7 < size; i += 8) {
        __m256 b_grad = _mm256_loadu_ps(b->grad->ptr + i);
        __m256 a_data = _mm256_loadu_ps(a->data->ptr + i);
        __m256 db = _mm256_fmadd_ps(dout_vec, a_data, b_grad);
        _mm256_storeu_ps(b->grad->ptr + i, db);
      }

      for (; i < size; ++i) {
        b->grad->ptr[i] += dout * a->data->ptr[i];
      }
    }
  }
}

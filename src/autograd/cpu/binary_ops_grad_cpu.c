#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#include "autograd/autograd.h"
#include "ops/ops.h"

void add_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  IDRAK_DEBUG("GRAD ", "add_grad_op: Computing gradient for addition\n");

  // Error checking for null tensors
  if (!out || !out->grad || !prev) {
    IDRAK_ERROR(
        "add_grad_op ERROR: Output tensor, output gradient, or previous "
        "tensors array is NULL! out=%p, out->grad=%p, prev=%p\n",
        (void *)out, (void *)out->grad, (void *)prev);
    return;
  }

  if (n_prev != 1 && n_prev != 2) {
    IDRAK_ERROR("add_grad_op ERROR: Invalid number of previous tensors: %d. "
                "Expected 1 or 2.\n",
                n_prev);
    return;
  }

  if (n_prev == 2) {
    if (!prev[0] || !prev[1]) {
      IDRAK_ERROR("add_grad_op ERROR: One or both previous tensors are NULL "
                  "when n_prev is 2! prev[0]=%p, prev[1]=%p\n",
                  (void *)prev[0], (void *)prev[1]);
      return;
    }
    if (prev[0]->requires_grad && !prev[0]->grad) {
      IDRAK_ERROR("add_grad_op ERROR: Previous tensor 0 requires grad but its "
                  "grad is NULL!\n");
      return;
    }
    if (prev[1]->requires_grad && !prev[1]->grad) {
      IDRAK_ERROR("add_grad_op ERROR: Previous tensor 1 requires grad but its "
                  "grad is NULL!\n");
      return;
    }
  } else if (n_prev == 1) {
    if (!prev[0]) {
      IDRAK_ERROR("add_grad_op ERROR: Previous tensor is NULL when n_prev is "
                  "1! prev[0]=%p\n",
                  (void *)prev[0]);
      return;
    }
    if (prev[0]->requires_grad && !prev[0]->grad) {
      IDRAK_ERROR("add_grad_op ERROR: Previous tensor 0 requires grad but its "
                  "grad is NULL!\n");
      return;
    }
    if (!extras) {
      IDRAK_ERROR("add_grad_op ERROR: Extras is NULL when n_prev is 1 (scalar "
                  "addition)!\n");
      return;
    }
  }

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
  IDRAK_DEBUG("GRAD ", "sub_grad_op: Computing gradient for subtraction\n");

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

void rsub_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  IDRAK_DEBUG("GRAD ", "rsub_grad_op: Computing gradient for reverse "
                       "subtraction\n");

  // Error checking for null tensors and invalid n_prev
  if (!out || !out->grad || !prev) {
    IDRAK_ERROR(
        "rsub_grad_op ERROR: Output tensor, output gradient, or previous "
        "tensors array is NULL! out=%p, out->grad=%p, prev=%p\n",
        (void *)out, (void *)out->grad, (void *)prev);
    return;
  }

  if (n_prev != 1) {
    IDRAK_ERROR("rsub_grad_op ERROR: Invalid number of previous tensors: %d. "
                "Expected 1.\n",
                n_prev);
    return;
  }

  if (!prev[0]) {
    IDRAK_ERROR("rsub_grad_op ERROR: Previous tensor is NULL! prev[0]=%p\n",
                (void *)prev[0]);
    return;
  }

  if (!extras) {
    IDRAK_ERROR("rsub_grad_op ERROR: Extras is NULL (scalar value missing)!\n");
    return;
  }

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
  IDRAK_DEBUG("GRAD ", "mul_grad_op: Computing gradient for multiplication\n");

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
          a->grad->ptr[a_offset] +=
              out->grad->ptr[out_offset] * b->data->ptr[b_offset];
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
          b->grad->ptr[b_offset] +=
              out->grad->ptr[out_offset] * a->data->ptr[a_offset];
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
  IDRAK_DEBUG("GRAD ", "pow_grad_op: Computing gradient for power operation\n");

  int size = numel(out->shape, out->ndim);
  int ndim = out->ndim;
  int *shape = out->shape;

  Tensor *a = prev[0];
  float b = *((float *)extras);

  if (!a->requires_grad)
    return;

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
  IDRAK_DEBUG("GRAD ", "div_grad_op: Computing gradient for division\n");

  // Basic null pointer checks
  if (!out || !prev) {
    IDRAK_ERROR("div_grad_op ERROR: Output tensor or previous tensors array "
                "is NULL! out=%p, prev=%p\n",
                (void *)out, (void *)prev);
    return;
  }

  if (!out->grad || !out->grad->ptr) {
    IDRAK_ERROR("div_grad_op ERROR: Output gradient is NULL! out->grad=%p\n",
                (void *)out->grad);
    return;
  }

  if (n_prev != 2 && n_prev != 1) {
    IDRAK_ERROR("div_grad_op ERROR: Invalid number of previous tensors: %d. "
                "Expected 1 or 2.\n",
                n_prev);
    return;
  }

  if (n_prev == 2) {
    if (!prev[0] || !prev[1]) {
      IDRAK_ERROR("div_grad_op ERROR: One or both previous tensors are NULL! "
                  "prev[0]=%p, prev[1]=%p\n",
                  (void *)prev[0], (void *)prev[1]);
      return;
    }
    if (!prev[0]->data || !prev[0]->data->ptr || !prev[1]->data ||
        !prev[1]->data->ptr) {
      IDRAK_ERROR("div_grad_op ERROR: One or both previous tensors' data are "
                  "NULL!\n");
      return;
    }
    if (prev[0]->requires_grad && (!prev[0]->grad || !prev[0]->grad->ptr)) {
      IDRAK_ERROR("div_grad_op ERROR: Previous tensor 0 requires grad but its "
                  "grad is NULL!\n");
      return;
    }
    if (prev[1]->requires_grad && (!prev[1]->grad || !prev[1]->grad->ptr)) {
      IDRAK_ERROR("div_grad_op ERROR: Previous tensor 1 requires grad but its "
                  "grad is NULL!\n");
      return;
    }
  } else if (n_prev == 1) {
    if (!prev[0]) {
      IDRAK_ERROR("div_grad_op ERROR: Previous tensor is NULL! prev[0]=%p\n",
                  (void *)prev[0]);
      return;
    }
    if (!extras) {
      IDRAK_ERROR(
          "div_grad_op ERROR: Extras is NULL (scalar value missing)!\n");
      return;
    }
    if (!prev[0]->data || !prev[0]->data->ptr) {
      IDRAK_ERROR("div_grad_op ERROR: Previous tensor's data is NULL!\n");
      return;
    }
    if (prev[0]->requires_grad && (!prev[0]->grad || !prev[0]->grad->ptr)) {
      IDRAK_ERROR("div_grad_op ERROR: Previous tensor requires grad but its "
                  "grad is NULL!\n");
      return;
    }
  }

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
          a->grad->ptr[a_offset] +=
              out->grad->ptr[out_offset] / b->data->ptr[b_offset];
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
          float b_val = b->data->ptr[b_offset];
          if (b_val != 0.0f) { // Add check for division by zero
            b->grad->ptr[b_offset] -= out->grad->ptr[out_offset] *
                                      a->data->ptr[a_offset] / (b_val * b_val);
          } else {
            // Handle division by zero: set gradient to 0 to prevent crash.
            b->grad->ptr[b_offset] = 0.0f;
          }
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
          float b_val = b->data->ptr[i];
          if (b_val != 0.0f) {
            a->grad->ptr[i] += out->grad->ptr[i] / b_val;
          } else {
            a->grad->ptr[i] = 0.0f;
          }
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

          // Handle division by zero for b_squared
          __m256 zero = _mm256_setzero_ps();
          __m256 b_squared_is_zero = _mm256_cmp_ps(b_squared, zero, _CMP_EQ_OQ);
          // Replace zero b_squared values with a small epsilon to avoid
          // division by zero
          b_squared = _mm256_blendv_ps(b_squared, _mm256_set1_ps(1e-8f),
                                       b_squared_is_zero);

          __m256 db =
              _mm256_fnmadd_ps(_mm256_div_ps(a_data, b_squared), dout, b_grad);
          _mm256_storeu_ps(b->grad->ptr + i, db);
        }

        for (; i < size; ++i) {
          float b_val = b->data->ptr[i];
          if (b_val != 0.0f) {
            b->grad->ptr[i] -=
                out->grad->ptr[i] * a->data->ptr[i] / (b_val * b_val);
          } else {
            b->grad->ptr[i] = 0.0f;
          }
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
  IDRAK_DEBUG("GRAD ",
              "rdiv_grad_op: Computing gradient for reverse division\n");

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
        a->grad->ptr[a_offset] +=
            out->grad->ptr[out_offset] * (-b) /
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
        a->grad->ptr[i] +=
            out->grad->ptr[i] * (-b) / (a->data->ptr[i] * a->data->ptr[i]);
      }
    }
  }
}

void matmul_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  IDRAK_DEBUG("GRAD ", "matmul_grad_op: Computing gradient for matrix "
                       "multiplication\n");
  if (!out || !prev) {
    IDRAK_ERROR("matmul_grad_op ERROR: Output tensor or previous tensors array "
                "is NULL! out=%p, prev=%p\n",
                (void *)out, (void *)prev);
    return;
  }

  if (!out->grad || !out->grad->ptr) {
    IDRAK_ERROR("matmul_grad_op ERROR: Output gradient is NULL! out->grad=%p\n",
                (void *)out->grad);
    return;
  }

  if (n_prev != 2) {
    IDRAK_ERROR("matmul_grad_op ERROR: Invalid number of previous tensors: %d. "
                "Expected 2.\n",
                n_prev);
    return;
  }

  if (!prev[0] || !prev[1]) {
    IDRAK_ERROR("matmul_grad_op ERROR: One or both previous tensors are NULL! "
                "prev[0]=%p, prev[1]=%p\n",
                (void *)prev[0], (void *)prev[1]);
    return;
  }

  Tensor *a = prev[0];
  Tensor *b = prev[1];

  if (a->ndim < 2 || b->ndim < 2 || out->ndim < 2) {
    IDRAK_ERROR("matmul_grad_op ERROR: All tensors must have at least 2 "
                "dimensions! a->ndim=%d, b->ndim=%d, out->ndim=%d\n",
                a->ndim, b->ndim, out->ndim);
    return;
  }

  // Validate shape arrays exist
  if (!a->shape || !b->shape || !out->shape) {
    IDRAK_ERROR("matmul_grad_op ERROR: One or more shape arrays are NULL!\n");
    return;
  }

  // Validate strides arrays exist
  if (!a->strides || !b->strides || !out->strides) {
    IDRAK_ERROR("matmul_grad_op ERROR: One or more stride arrays are NULL!\n");
    return;
  }

  // Validate data arrays exist
  if (!a->data || !a->data->ptr || !b->data || !b->data->ptr) {
    IDRAK_ERROR("matmul_grad_op ERROR: One or more data arrays are NULL!\n");
    return;
  }

  // Now safe to access dimensions
  int N = a->shape[a->ndim - 2]; // rows of a
  int K = a->shape[a->ndim - 1]; // cols of a / rows of b
  int M = b->shape[b->ndim - 1]; // cols of b

  // Dimension compatibility checks
  if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) {
    IDRAK_ERROR(
        "matmul_grad_op ERROR: Dimension mismatch for matrix multiplication! "
        "a->shape[last]=%d, b->shape[second_last]=%d\n",
        a->shape[a->ndim - 1], b->shape[b->ndim - 2]);
    return;
  }

  // Validate output dimensions match expected result
  if (out->shape[out->ndim - 2] != N || out->shape[out->ndim - 1] != M) {
    IDRAK_ERROR(
        "matmul_grad_op ERROR: Output dimensions don't match expected result! "
        "Expected (%d, %d), got (%d, %d)\n",
        N, M, out->shape[out->ndim - 2], out->shape[out->ndim - 1]);
    return;
  }

  // Calculate batch dimensions - use the maximum batch dimensions among all
  // tensors
  int max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
  max_ndim = (max_ndim > out->ndim) ? max_ndim : out->ndim;

  // Calculate total batch size (product of all batch dimensions)
  int batch_size = 1;
  for (int i = 0; i < max_ndim - 2; ++i) {
    int a_dim = (i < a->ndim - 2) ? a->shape[i] : 1;
    int b_dim = (i < b->ndim - 2) ? b->shape[i] : 1;
    int out_dim = (i < out->ndim - 2) ? out->shape[i] : 1;

    // Verify broadcasting compatibility
    if ((a_dim != 1 && b_dim != 1 && a_dim != b_dim) ||
        (a_dim != 1 && out_dim != 1 && a_dim != out_dim) ||
        (b_dim != 1 && out_dim != 1 && b_dim != out_dim)) {
      IDRAK_ERROR(
          "matmul_grad_op ERROR: Incompatible batch dimensions at index %d: "
          "a=%d, b=%d, out=%d\n",
          i, a_dim, b_dim, out_dim);
      return;
    }

    batch_size *= out_dim;
  }

  // Calculate strides for matrix operations (last two dimensions)
  int a_row_stride = a->strides[a->ndim - 2];
  int a_col_stride = a->strides[a->ndim - 1];
  int b_row_stride = b->strides[b->ndim - 2];
  int b_col_stride = b->strides[b->ndim - 1];
  int out_row_stride = out->strides[out->ndim - 2];
  int out_col_stride = out->strides[out->ndim - 1];

  // Calculate total matrix sizes for batch indexing
  int a_matrix_size = N * K;
  int b_matrix_size = K * M;
  int out_matrix_size = N * M;

  // Compute gradient for tensor a: grad_a += out_grad @ b^T
  if (a->requires_grad) {
    if (!a->grad || !a->grad->ptr) {
      IDRAK_ERROR("matmul_grad_op ERROR: Tensor 'a' requires grad but its grad "
                  "is NULL!\n");
      return;
    }

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      // Calculate batch offsets with proper broadcasting
      int a_batch_offset = 0;
      int b_batch_offset = 0;
      int out_batch_offset = 0;

      int temp_batch_idx = batch_idx;
      for (int dim = max_ndim - 3; dim >= 0; --dim) {
        int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
        int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
        int out_dim = (dim < out->ndim - 2) ? out->shape[dim] : 1;

        int coord = temp_batch_idx % out_dim;
        temp_batch_idx /= out_dim;

        if (dim < a->ndim - 2 && a_dim > 1) {
          a_batch_offset += coord * a->strides[dim];
        }
        if (dim < b->ndim - 2 && b_dim > 1) {
          b_batch_offset += coord * b->strides[dim];
        }
        if (dim < out->ndim - 2) {
          out_batch_offset += coord * out->strides[dim];
        }
      }

      // Compute grad_a[i,j] += sum_k(out_grad[i,k] * b[j,k])
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
          float sum = 0.0f;
          for (int k = 0; k < M; ++k) {
            float out_grad_val =
                out->grad->ptr[out_batch_offset + i * out_row_stride +
                               k * out_col_stride];
            float b_val =
                b->data
                    ->ptr[b_batch_offset + j * b_row_stride + k * b_col_stride];
            sum += out_grad_val * b_val;
          }
          a->grad->ptr[a_batch_offset + i * a_row_stride + j * a_col_stride] +=
              sum;
        }
      }
    }
  }

  // Compute gradient for tensor b: grad_b += a^T @ out_grad
  if (b->requires_grad) {
    if (!b->grad || !b->grad->ptr) {
      IDRAK_ERROR("matmul_grad_op ERROR: Tensor 'b' requires grad but its grad "
                  "is NULL!\n");
      return;
    }

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      // Calculate batch offsets with proper broadcasting
      int a_batch_offset = 0;
      int b_batch_offset = 0;
      int out_batch_offset = 0;

      int temp_batch_idx = batch_idx;
      for (int dim = max_ndim - 3; dim >= 0; --dim) {
        int a_dim = (dim < a->ndim - 2) ? a->shape[dim] : 1;
        int b_dim = (dim < b->ndim - 2) ? b->shape[dim] : 1;
        int out_dim = (dim < out->ndim - 2) ? out->shape[dim] : 1;

        int coord = temp_batch_idx % out_dim;
        temp_batch_idx /= out_dim;

        if (dim < a->ndim - 2 && a_dim > 1) {
          a_batch_offset += coord * a->strides[dim];
        }
        if (dim < b->ndim - 2 && b_dim > 1) {
          b_batch_offset += coord * b->strides[dim];
        }
        if (dim < out->ndim - 2) {
          out_batch_offset += coord * out->strides[dim];
        }
      }

      // Compute grad_b[i,j] += sum_k(a[k,i] * out_grad[k,j])
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
          float sum = 0.0f;
          for (int k = 0; k < N; ++k) {
            float a_val =
                a->data
                    ->ptr[a_batch_offset + k * a_row_stride + i * a_col_stride];
            float out_grad_val =
                out->grad->ptr[out_batch_offset + k * out_row_stride +
                               j * out_col_stride];
            sum += a_val * out_grad_val;
          }
          b->grad->ptr[b_batch_offset + i * b_row_stride + j * b_col_stride] +=
              sum;
        }
      }
    }
  }

  IDRAK_DEBUG("GRAD ", "matmul_grad_op: Gradient computation completed "
                       "successfully\n");
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
  IDRAK_DEBUG("GRAD ",
              "conv2d_grad_op: Computing gradient for 2D convolution\n");

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
  IDRAK_DEBUG("GRAD ", "dot_grad_op: Computing gradient for dot product\n");

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

#include "ops.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>

void add_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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
      int size = numel(a->shape, a->ndim);
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
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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

void sub_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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
      int size = numel(a->shape, a->ndim);
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
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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

void rsub_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  float b = *((float *)extras);

  if (a->requires_grad) {
    int i = 0;
    int size = numel(a->shape, a->ndim);
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

void mul_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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
      int size = numel(a->shape, a->ndim);
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
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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

void div_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  if (n_prev == 2) {
    Tensor *a = prev[0];
    Tensor *b = prev[1];

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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
      int size = numel(a->shape, a->ndim);
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
  } else if (n_prev == 1 && extras != NULL) {
    Tensor *a = prev[0];
    float b = *((float *)extras);

    if (a->requires_grad) {
      int i = 0;
      int size = numel(a->shape, a->ndim);
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

void rdiv_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  Tensor *a = prev[0];
  float b = *((float *)extras);

  if (a->requires_grad) {
    int i = 0;
    int size = numel(a->shape, a->ndim);
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

#include "ops/ops.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>
#include "logger.h"

static void reconfigure_scalar_output(Tensor *in_tensor, Tensor *out) {
  LOG_INFO("OP: reconfigure_scalar_output: Reconfiguring output "
                       "tensor for scalar op");
  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides &&
      out->ndim > 0) { // compute_strides can return NULL if ndim=0 or error
    LOG_ERROR("reconfigure_scalar_output: Failed to allocate memory for "
                "out->strides.");
    free(out->shape);
    out->shape = NULL;
    return;
  }

  size_t out_total_size = numel(out->shape, out->ndim);
  if (!out->data->elems) {
    LOG_ERROR("reconfigure_scalar_output: Failed to allocate memory for "
                "out->data->elems.");
    free(out->shape);
    out->shape = NULL;
    if (out->strides) {
      free(out->strides);
      out->strides = NULL;
    }
    return;
  }
}

void add_scalar_op(Tensor *a, float b, Tensor *out) {
  LOG_INFO("OP: add_scalar_op: Performing scalar addition (scalar=%.2f)", b);

  reconfigure_scalar_output(a, out);
  if (!out->data->elems) {
    return;
  }

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

      ((float*)out->data->elems)[offset_out] = ((float*)a->data->elems)[offset_a] + b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(((float*)a->data->elems) + i);
      __m256 z = _mm256_add_ps(x, scalar);
      _mm256_storeu_ps(((float*)out->data->elems) + i, z);
    }

    for (; i < size; ++i) {
      ((float*)out->data->elems)[i] = ((float*)a->data->elems)[i] + b;
    }
  }
}

void sub_scalar_op(Tensor *a, float b, Tensor *out) {
  LOG_INFO("OP: sub_scalar_op: Performing scalar subtraction (scalar=%.2f)", b);

  reconfigure_scalar_output(a, out);
  if (!out->data->elems) {
    return;
  }

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

      ((float*)out->data->elems)[offset_out] = ((float*)a->data->elems)[offset_a] - b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(((float*)a->data->elems) + i);
      __m256 z = _mm256_sub_ps(x, scalar);
      _mm256_storeu_ps(((float*)out->data->elems) + i, z);
    }

    for (; i < size; ++i) {
      ((float*)out->data->elems)[i] = ((float*)a->data->elems)[i] - b;
    }
  }
}

void rsub_scalar_op(float a, Tensor *b, Tensor *out) {
  LOG_INFO("OP: rsub_scalar_op: Performing reverse scalar subtraction (scalar=%.2f)", a);

  // Error checking for null tensors
  if (!b || !out) {
    LOG_ERROR(
        "rsub_scalar_op ERROR: Input or output tensor is NULL! b=%p, out=%p",
        (void *)b, (void *)out);
    return;
  }

  reconfigure_scalar_output(b, out);
  if (!out->data->elems) {
    LOG_ERROR("rsub_scalar_op ERROR: Output tensor data pointer is NULL "
                "after reconfiguration.");
    return;
  }

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

      ((float*)out->data->elems)[out_offset] = a - ((float*)b->data->elems)[b_offset];
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(a);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(((float*)b->data->elems) + i);
      __m256 z = _mm256_sub_ps(scalar, x);
      _mm256_storeu_ps(((float*)out->data->elems) + i, z);
    }

    for (; i < size; ++i) {
      ((float*)out->data->elems)[i] = a - ((float*)b->data->elems)[i];
    }
  }
}

void mul_scalar_op(Tensor *a, float b, Tensor *out) {
  LOG_INFO("OP: mul_scalar_op: Performing scalar multiplication (scalar=%.2f)", b);

  reconfigure_scalar_output(a, out);
  if (!out->data->elems) {
    return;
  }

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

      ((float*)out->data->elems)[offset_out] = ((float*)a->data->elems)[offset_a] * b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(((float*)a->data->elems) + i);
      __m256 z = _mm256_mul_ps(x, scalar);
      _mm256_storeu_ps(((float*)out->data->elems) + i, z);
    }

    for (; i < size; ++i) {
      ((float*)out->data->elems)[i] = ((float*)a->data->elems)[i] * b;
    }
  }
}

void div_scalar_op(Tensor *a, float b, Tensor *out) {
  LOG_INFO("OP: div_scalar_op: Performing scalar division (scalar=%.2f)", b);

  reconfigure_scalar_output(a, out);
  if (!out->data->elems) {
    return;
  }

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

      if (b == 0.0f) {
        LOG_WARN("Division by zero in div_scalar_op at index %d. "
                      "Result will be +/-INF or NaN.",
                      idx);
        ((float*)out->data->elems)[offset_out] =
            ((float*)a->data->elems)[offset_a] / b; // Will result in INF/NaN
      } else {
        ((float*)out->data->elems)[offset_out] = ((float*)a->data->elems)[offset_a] / b;
      }
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);
    if (b == 0.0f) {
      LOG_WARN("Division by zero in div_scalar_op (SIMD path). "
                    "Results will be +/-INF or NaN.");
    }

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(((float*)a->data->elems) + i);
      __m256 z = _mm256_div_ps(x, scalar);
      _mm256_storeu_ps(((float*)out->data->elems) + i, z);
    }

    for (; i < size; ++i) {
      if (b == 0.0f) {
        ((float*)out->data->elems)[i] = ((float*)a->data->elems)[i] / b;
      } else {
        ((float*)out->data->elems)[i] = ((float*)a->data->elems)[i] / b;
      }
    }
  }
}

void rdiv_scalar_op(Tensor *a, float b, Tensor *out) {
  LOG_INFO("OP: rdiv_scalar_op: Performing reverse scalar division (scalar=%.2f)", b);

  reconfigure_scalar_output(a, out);
  if (!out->data->elems) {
    return;
  }

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

      if (((float*)a->data->elems)[offset_a] == 0.0f) {
        LOG_WARN("Division by zero in rdiv_scalar_op at index %d. "
                      "Result will be +/-INF or NaN.",
                      idx);
        ((float*)out->data->elems)[offset_out] =
            b / ((float*)a->data->elems)[offset_a]; // Will result in INF/NaN
      } else {
        ((float*)out->data->elems)[offset_out] = b / ((float*)a->data->elems)[offset_a];
      }
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(((float*)a->data->elems) + i);
      __m256 z = _mm256_div_ps(scalar, x);
      _mm256_storeu_ps(((float*)out->data->elems) + i, z);
    }

    for (; i < size; ++i) {
      if (((float*)a->data->elems)[i] == 0.0f) {
        LOG_WARN("Division by zero in rdiv_scalar_op at index %d. "
                      "Result will be +/-INF or NaN.",
                      i);
        ((float*)out->data->elems)[i] = b / ((float*)a->data->elems)[i];
      } else {
        ((float*)out->data->elems)[i] = b / ((float*)a->data->elems)[i];
      }
    }
  }
}

void pow_scalar_op(Tensor *a, float b, Tensor *out) {
  LOG_INFO("OP: pow_scalar_op: Performing scalar power (exponent=%.2f)", b);

  reconfigure_scalar_output(a, out);
  if (!out->data->elems) {
    return;
  }

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

      ((float*)out->data->elems)[offset_out] = powf(((float*)a->data->elems)[offset_a], b);
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(((float*)a->data->elems) + i);
      __m256 y = Sleef_powf8_u10avx2(x, scalar);
      _mm256_storeu_ps(((float*)out->data->elems) + i, y);
    }

    for (; i < size; ++i) {
      ((float*)out->data->elems)[i] = powf(((float*)a->data->elems)[i], b);
    }
  }
}
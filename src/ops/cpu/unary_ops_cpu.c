#include "ops/ops.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#if DEBUG
#include "utils.h"
#endif

static void reconfigure_unary_output(Tensor *in, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] reconfigure_unary_output: Reconfiguring output "
              "tensor for unary op\n");

  if (out->shape) {
    free(out->shape);
    out->shape = NULL;
  }
  if (out->strides) {
    free(out->strides);
    out->strides = NULL;
  }
  if (out->data && out->data->ptr) {
    free(out->data->ptr);
    out->data->ptr = NULL;
  }

  out->ndim = in->ndim;
  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    return;
  }
  memcpy(out->shape, in->shape, out->ndim * sizeof(int));

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides && out->ndim > 0) {
    fprintf(stderr, "Error: Failed to allocate memory for out->strides\n");
    free(out->shape);
    out->shape = NULL;
    return;
  }

  size_t out_total_size = numel(out->shape, out->ndim);
  out->data->ptr = (float *)malloc(out_total_size * sizeof(float));
  if (!out->data->ptr) {
    fprintf(stderr, "Error: Failed to allocate memory for out->data->ptr\n");
    free(out->shape);
    out->shape = NULL;
    if (out->strides) {
      free(out->strides);
      out->strides = NULL;
    }
    return;
  }
}

void relu_op(Tensor *in, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] relu_op: Performing ReLU activation\n");

  reconfigure_unary_output(in, out);
  if (!out->data->ptr) {
    return;
  }

  int size = numel(in->shape, in->ndim);

  if (!is_contiguous(in) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_in = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = in->ndim - 1; d >= 0; --d) {
        int coord = tmp % in->shape[d];
        tmp /= in->shape[d];
        offset_in += coord * in->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] =
          in->data->ptr[offset_in] > 0 ? in->data->ptr[offset_in] : 0.0f;
    }
  } else {
    int i = 0;
    __m256 zeros = _mm256_setzero_ps();

    for (; i + 7 < size; i += 8) {
      __m256 vin = _mm256_loadu_ps(in->data->ptr + i);
      __m256 vout = _mm256_max_ps(vin, zeros);
      _mm256_storeu_ps(out->data->ptr + i, vout);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = in->data->ptr[i] > 0.0f ? in->data->ptr[i] : 0.0f;
    }
  }

  out->requires_grad = in->requires_grad;
}

void log_op(Tensor *in, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] log_op: Performing natural logarithm\n");

  reconfigure_unary_output(in, out);
  if (!out->data->ptr) {
    return;
  }

  int size = numel(in->shape, in->ndim);

  if (!is_contiguous(in) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_in = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = in->ndim - 1; d >= 0; --d) {
        int coord = tmp % in->shape[d];
        tmp /= in->shape[d];
        offset_in += coord * in->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = logf(in->data->ptr[offset_in]);
    }
  } else {
    int i = 0;

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(in->data->ptr + i);
      __m256 z = Sleef_logf8_u10avx2(x);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = logf(in->data->ptr[i]);
    }
  }

  out->requires_grad = in->requires_grad ? true : false;
}

void exp_op(Tensor *in, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] exp_op: Performing exponential\n");

  reconfigure_unary_output(in, out);
  if (!out->data->ptr) {
    return;
  }

  int size = numel(in->shape, in->ndim);

  if (!is_contiguous(in) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_in = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = in->ndim - 1; d >= 0; --d) {
        int coord = tmp % in->shape[d];
        tmp /= in->shape[d];
        offset_in += coord * in->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = expf(in->data->ptr[offset_in]);
    }
  } else {
    int i = 0;

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(in->data->ptr + i);
      __m256 z = Sleef_expf8_u10avx2(x);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = expf(in->data->ptr[i]);
    }
  }

  out->requires_grad = in->requires_grad ? true : false;
}

void softmax_op(Tensor *in, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] softmax_op: Performing softmax activation\n");
}

void neg_op(Tensor *in, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] neg_op: Performing negation\n");

  reconfigure_unary_output(in, out);
  if (!out->data->ptr) {
    return;
  }

  int size = numel(in->shape, in->ndim);

  if (!is_contiguous(in) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_in = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = in->ndim - 1; d >= 0; --d) {
        int coord = tmp % in->shape[d];
        tmp /= in->shape[d];
        offset_in += coord * in->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = 0.0f - in->data->ptr[offset_in];
    }

  } else {
    int i = 0;

    __m256 zeros = _mm256_setzero_ps();

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(in->data->ptr + i);
      __m256 y = _mm256_sub_ps(zeros, x);
      _mm256_storeu_ps(out->data->ptr + i, y);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = 0.0f - in->data->ptr[i];
    }
  }

  out->requires_grad = in->requires_grad;
}

void abs_op(Tensor *in, Tensor *out) {

  DEBUG_PRINT("[IDRAK_DEBUG] abs_op: Performing absolute value\n");

  reconfigure_unary_output(in, out);
  if (!out->data->ptr) {
    return;
  }

  int size = numel(in->shape, in->ndim);

  if (!is_contiguous(in) || !is_contiguous(out)) {
    for (int idx = 0; idx < size; ++idx) {
      int offset_in = 0;
      int offset_out = 0;
      int tmp = idx;

      for (int d = in->ndim - 1; d >= 0; --d) {
        int coord = tmp % in->shape[d];
        tmp /= in->shape[d];
        offset_in += coord * in->strides[d];
        offset_out += coord * out->strides[d];
      }

      out->data->ptr[offset_out] = in->data->ptr[offset_in] >= 0.0f
                                       ? in->data->ptr[offset_in]
                                       : 0.0f - in->data->ptr[offset_in];
    }
  } else {
    int i = 0;
    __m256 mask = _mm256_castsi256_ps(
        _mm256_set1_epi32(0x7FFFFFFF)); // mask to remove sign bit

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(in->data->ptr + i);
      __m256 y = _mm256_and_ps(x, mask);
      _mm256_storeu_ps(out->data->ptr + i, y);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] =
          in->data->ptr[i] >= 0 ? in->data->ptr[i] : 0.0f - in->data->ptr[i];
    }
  }

  out->requires_grad = in->requires_grad;
}

#include "ops/ops.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#if DEBUG
#include "utils.h"
#endif

void relu_op(Tensor *in, Tensor *out) {
  IDRAK_DEBUG("OP   ", "relu_op: Performing ReLU activation\n");

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
}

void log_op(Tensor *in, Tensor *out) {
  IDRAK_DEBUG("OP   ", "log_op: Performing natural logarithm\n");

  if (!in || !out) {
    IDRAK_ERROR("log_op ERROR: Input or output tensor is NULL! in=%p, out=%p\n",
                (void *)in, (void *)out);
    return;
  }

  if (!out->data->ptr) {
    IDRAK_ERROR("log_op ERROR: Output tensor data pointer is NULL after "
                "reconfiguration.\n");
    return;
  }

  int size = numel(in->shape, in->ndim);

  for (int i = 0; i < size; ++i) {
    if (in->data->ptr[i] < 0.0f) {
      IDRAK_ERROR("log_op ERROR: Input value at index %d is negative (%.4f)! "
                  "Logarithm of negative number is undefined.\n",
                  i, in->data->ptr[i]);
    } else if (in->data->ptr[i] == 0.0f) {
      IDRAK_WARNING("log_op WARNING: Input value at index %d is zero. "
                    "Logarithm of zero is -INF.\n",
                    i);
    }
  }

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
}

void exp_op(Tensor *in, Tensor *out) {
  IDRAK_DEBUG("OP   ", "exp_op: Performing exponential\n");

  // Error checking for null tensors
  if (!in || !out) {
    IDRAK_ERROR("exp_op ERROR: Input or output tensor is NULL! in=%p, out=%p\n",
                (void *)in, (void *)out);
    return;
  }

  if (!out->data->ptr) {
    IDRAK_ERROR("exp_op ERROR: Output tensor data pointer is NULL after "
                "reconfiguration.\n");
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
}

void softmax_op(Tensor *in, Tensor *out) {
  IDRAK_DEBUG("OP   ", "softmax_op: Performing softmax activation\n");
}

void neg_op(Tensor *in, Tensor *out) {
  IDRAK_DEBUG("OP   ", "neg_op: Performing negation\n");

  // Error checking for null tensors
  if (!in || !out) {
    IDRAK_ERROR("neg_op ERROR: Input or output tensor is NULL! in=%p, out=%p\n",
                (void *)in, (void *)out);
    return;
  }

  if (!out->data->ptr) {
    IDRAK_ERROR("neg_op ERROR: Output tensor data pointer is NULL after "
                "reconfiguration.\n");
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
}

void abs_op(Tensor *in, Tensor *out) {

  IDRAK_DEBUG("OP   ", "abs_op: Performing absolute value\n");

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
}

#include "ops/ops.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

static void reconfigure_scalar_output(Tensor *in_tensor, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] reconfigure_scalar_output: Reconfiguring output "
              "tensor for scalar op\n");

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

  out->ndim = in_tensor->ndim;
  out->shape = (int *)malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    fprintf(stderr, "Error: Failed to allocate memory for out->shape\n");
    return;
  }
  memcpy(out->shape, in_tensor->shape, out->ndim * sizeof(int));

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides &&
      out->ndim > 0) { // compute_strides can return NULL if ndim=0 or error
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

void add_scalar_op(Tensor *a, float b, Tensor *out) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] add_scalar_op: Performing scalar addition (scalar=%.2f)\n",
      b);

  reconfigure_scalar_output(a, out);
  if (!out->data->ptr) {
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

      out->data->ptr[offset_out] = a->data->ptr[offset_a] + b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_add_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] + b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void sub_scalar_op(Tensor *a, float b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] sub_scalar_op: Performing scalar subtraction "
              "(scalar=%.2f)\n",
              b);

  reconfigure_scalar_output(a, out);
  if (!out->data->ptr) {
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

      out->data->ptr[offset_out] = a->data->ptr[offset_a] - b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_sub_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] - b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void rsub_scalar_op(float a, Tensor *b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] rsub_scalar_op: Performing reverse scalar "
              "subtraction (scalar=%.2f)\n",
              a);

  reconfigure_scalar_output(b, out);
  if (!out->data->ptr) {
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

      out->data->ptr[out_offset] = a - b->data->ptr[b_offset];
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(a);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(b->data->ptr + i);
      __m256 z = _mm256_sub_ps(scalar, x);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a - b->data->ptr[i];
    }
  }

  out->requires_grad = b->requires_grad ? true : false;
}

void mul_scalar_op(Tensor *a, float b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] mul_scalar_op: Performing scalar multiplication "
              "(scalar=%.2f)\n",
              b);

  reconfigure_scalar_output(a, out);
  if (!out->data->ptr) {
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

      out->data->ptr[offset_out] = a->data->ptr[offset_a] * b;
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_mul_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = a->data->ptr[i] * b;
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void div_scalar_op(Tensor *a, float b, Tensor *out) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] div_scalar_op: Performing scalar division (scalar=%.2f)\n",
      b);

  reconfigure_scalar_output(a, out);
  if (!out->data->ptr) {
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
        fprintf(stderr,
                "Warning: Division by zero in div_scalar_op at index %d. "
                "Result will be +/-INF or NaN.\n",
                idx);
        out->data->ptr[offset_out] =
            a->data->ptr[offset_a] / b; // Will result in INF/NaN
      } else {
        out->data->ptr[offset_out] = a->data->ptr[offset_a] / b;
      }
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);
    if (b == 0.0f) {
      fprintf(stderr, "Warning: Division by zero in div_scalar_op (SIMD path). "
                      "Results will be +/-INF or NaN.\n");
    }

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_div_ps(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      if (b == 0.0f) {
        out->data->ptr[i] = a->data->ptr[i] / b;
      } else {
        out->data->ptr[i] = a->data->ptr[i] / b;
      }
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void rdiv_scalar_op(Tensor *a, float b, Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] rdiv_scalar_op: Performing reverse scalar "
              "division (scalar=%.2f)\n",
              b);

  reconfigure_scalar_output(a, out);
  if (!out->data->ptr) {
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

      if (a->data->ptr[offset_a] == 0.0f) {
        fprintf(stderr,
                "Warning: Division by zero in rdiv_scalar_op at index %d. "
                "Result will be +/-INF or NaN.\n",
                idx);
        out->data->ptr[offset_out] =
            b / a->data->ptr[offset_a]; // Will result in INF/NaN
      } else {
        out->data->ptr[offset_out] = b / a->data->ptr[offset_a];
      }
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 z = _mm256_div_ps(scalar, x);
      _mm256_storeu_ps(out->data->ptr + i, z);
    }

    for (; i < size; ++i) {
      if (a->data->ptr[i] == 0.0f) {
        fprintf(stderr,
                "Warning: Division by zero in rdiv_scalar_op at index %d. "
                "Result will be +/-INF or NaN.\n",
                i);
        out->data->ptr[i] = b / a->data->ptr[i];
      } else {
        out->data->ptr[i] = b / a->data->ptr[i];
      }
    }
  }

  out->requires_grad = a->requires_grad ? true : false;
}

void pow_scalar_op(Tensor *a, float b, Tensor *out) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] pow_scalar_op: Performing scalar power (exponent=%.2f)\n",
      b);

  reconfigure_scalar_output(a, out);
  if (!out->data->ptr) {
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

      out->data->ptr[offset_out] = powf(a->data->ptr[offset_a], b);
    }
  } else {
    int i = 0;
    __m256 scalar = _mm256_set1_ps(b);

    for (; i + 7 < size; i += 8) {
      __m256 x = _mm256_loadu_ps(a->data->ptr + i);
      __m256 y = Sleef_powf8_u10avx2(x, scalar);
      _mm256_storeu_ps(out->data->ptr + i, y);
    }

    for (; i < size; ++i) {
      out->data->ptr[i] = powf(a->data->ptr[i], b);
    }
  }
  out->requires_grad = a->requires_grad;
}

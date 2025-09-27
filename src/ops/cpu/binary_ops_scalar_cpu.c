#include "logger.h"
#include "ops/binary_scalar_ops.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#define SIMD_WIDTH 8

void add_scalar_op(Tensor* a, float b, Tensor* out) {
    LOG_INFO("OP: add_scalar_op: Performing scalar addition (scalar=%.2f)", b);

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

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

            data[offset_out] = a->data->data[offset_a] + b;
        }
    } else {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_add_ps(x, scalar);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            data[i] = a->data->data[i] + b;
        }
    }
    from_data(out, data);
    free(data);
}

void sub_scalar_op(Tensor* a, float b, Tensor* out) {
    LOG_INFO("OP: sub_scalar_op: Performing scalar subtraction (scalar=%.2f)", b);

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

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

            data[offset_out] = a->data->data[offset_a] - b;
        }
    } else {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_sub_ps(x, scalar);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            data[i] = a->data->data[i] - b;
        }
    }
    from_data(out, data);
    free(data);
}

void rsub_scalar_op(Tensor* a, float b, Tensor* out) {
    LOG_INFO("OP: rsub_scalar_op: Performing reverse scalar subtraction (scalar=%.2f)", b);

    // Error checking for null tensors
    if (!b || !out) {
        LOG_ERROR("rsub_scalar_op ERROR: Input or output tensor is NULL! b=%p, out=%p", (void*)a, (void*)out);
        return;
    }

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

    if (!is_contiguous(a) || !is_contiguous(out)) {
        for (int idx = 0; idx < size; ++idx) {
            int a_offset = 0;
            int out_offset = 0;
            int tmp = idx;

            for (int d = a->ndim - 1; d >= 0; --d) {
                int coord = tmp % a->shape[d];
                tmp /= a->shape[d];

                a_offset += coord * a->strides[d];
                out_offset += coord * out->strides[d];
            }

            data[out_offset] = b - a->data->data[a_offset];
        }
    } else {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_sub_ps(scalar, x);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            data[i] = b - a->data->data[i];
        }
    }
    from_data(out, data);
    free(data);
}

void mul_scalar_op(Tensor* a, float b, Tensor* out) {
    LOG_INFO("OP: mul_scalar_op: Performing scalar multiplication (scalar=%.2f)", b);

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

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

            data[offset_out] = a->data->data[offset_a] * b;
        }
    } else {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_mul_ps(x, scalar);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            data[i] = a->data->data[i] * b;
        }
    }
    from_data(out, data);
    free(data);
}

void div_scalar_op(Tensor* a, float b, Tensor* out) {
    LOG_INFO("OP: div_scalar_op: Performing scalar division (scalar=%.2f)", b);

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

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
                data[offset_out] = a->data->data[offset_a] / b; // Will result in INF/NaN
            } else {
                data[offset_out] = a->data->data[offset_a] / b;
            }
        }
    } else {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);
        if (b == 0.0f) {
            LOG_WARN("Division by zero in div_scalar_op (SIMD path). "
                     "Results will be +/-INF or NaN.");
        }

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_div_ps(x, scalar);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            if (b == 0.0f) {
                data[i] = a->data->data[i] / b;
            } else {
                data[i] = a->data->data[i] / b;
            }
        }
    }
    from_data(out, data);
    free(data);
}

void rdiv_scalar_op(Tensor* a, float b, Tensor* out) {
    LOG_INFO("OP: rdiv_scalar_op: Performing reverse scalar division (scalar=%.2f)", b);

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

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

            if (a->data->data[offset_a] == 0.0f) {
                LOG_WARN("Division by zero in rdiv_scalar_op at index %d. "
                         "Result will be +/-INF or NaN.",
                         idx);
                data[offset_out] = b / a->data->data[offset_a]; // Will result in INF/NaN
            } else {
                data[offset_out] = b / a->data->data[offset_a];
            }
        }
    } else {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 z = _mm256_div_ps(scalar, x);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            if (a->data->data[i] == 0.0f) {
                LOG_WARN("Division by zero in rdiv_scalar_op at index %d. "
                         "Result will be +/-INF or NaN.",
                         i);
                out->data->data[i] = b / a->data->data[i];
            } else {
                data[i] = b / a->data->data[i];
            }
        }
    }
    from_data(out, data);
    free(data);
}

void pow_scalar_op(Tensor* a, float b, Tensor* out) {
    LOG_INFO("OP: pow_scalar_op: Performing scalar power (exponent=%.2f)", b);

    int size = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * size);

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

            data[offset_out] = powf(a->data->data[offset_a], b);
        }
    } else {
        int i = 0;
        __m256 scalar = _mm256_set1_ps(b);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 y = Sleef_powf8_u10avx2(x, scalar);
            _mm256_storeu_ps(data + i, y);
        }

        for (; i < size; ++i) {
            data[i] = powf(a->data->data[i], b);
        }
    }
    from_data(out, data);
    free(data);
}

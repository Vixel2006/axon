#include "logger.h"
#include "ops/init_ops.h"
#include "ops/unary_ops.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#if DEBUG
#include "utils.h"
#endif

#define SIMD_WIDTH 8

#define COMPUTE_UNARY_OFFSETS(linear_idx, in_tensor, off_in, off_out)                                                                                                                                                                                                                                      \
    do {                                                                                                                                                                                                                                                                                                   \
        int idx = linear_idx;                                                                                                                                                                                                                                                                              \
        off_in = 0;                                                                                                                                                                                                                                                                                        \
        off_out = 0;                                                                                                                                                                                                                                                                                       \
        for (int d = in_tensor->ndim - 1; d >= 0; --d) {                                                                                                                                                                                                                                                   \
            int coord = idx % in_tensor->shape[d];                                                                                                                                                                                                                                                         \
            idx /= in_tensor->shape[d];                                                                                                                                                                                                                                                                    \
            off_in += coord * in_tensor->strides[d];                                                                                                                                                                                                                                                       \
            off_out += coord * out->strides[d];                                                                                                                                                                                                                                                            \
        }                                                                                                                                                                                                                                                                                                  \
    } while (0)

static inline bool check_tensors_unary(Tensor* in, Tensor* out, const char* op_name) {
    if (!in || !out) {
        LOG_ERROR("%s ERROR: NULL tensor! in=%p, out=%p", op_name, (void*)in, (void*)out);
        return false;
    }
    return true;
}

static inline float* alloc_tensor_data(int size, const char* op_name) {
    float* data = (float*)malloc(sizeof(float) * size);
    if (!data) {
        LOG_ERROR("%s ERROR: Failed to allocate memory for %d floats", op_name, size);
        return NULL;
    }
    return data;
}

static inline bool can_use_simd_unary(Tensor* in, Tensor* out) {
    return is_contiguous(in) && is_contiguous(out);
}

void relu_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: relu_op: Performing ReLU activation");

    if (!check_tensors_unary(in, out, "relu_op"))
        return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "relu_op");
    if (!data)
        return;

    if (!can_use_simd_unary(in, out)) {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear) {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = in->data->data[offset_in] > 0 ? in->data->data[offset_in] : 0.0f;
        }
    } else {
        int i = 0;
        __m256 zeros = _mm256_setzero_ps();

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 vin = _mm256_loadu_ps(in->data->data + i);
            __m256 vout = _mm256_max_ps(vin, zeros);
            _mm256_storeu_ps(data + i, vout);
        }

        for (; i < size; ++i) {
            data[i] = in->data->data[i] > 0.0f ? in->data->data[i] : 0.0f;
        }
    }
    from_data(out, data);
    free(data);
}

void log_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: log_op: Performing natural logarithm");

    if (!check_tensors_unary(in, out, "log_op"))
        return;

    int size = numel(in->shape, in->ndim);

    bool is_contig = is_contiguous(in);
    for (int linear = 0; linear < size; ++linear) {
        int offset_in;
        if (is_contig) {
            offset_in = linear;
        } else {
            int idx = linear;
            offset_in = 0;
            for (int d = in->ndim - 1; d >= 0; --d) {
                int coord = idx % in->shape[d];
                idx /= in->shape[d];
                offset_in += coord * in->strides[d];
            }
        }
        float val = in->data->data[offset_in];
        if (val < 0.0f) {
            LOG_ERROR("log_op ERROR: Input value at linear index %d is negative (%.4f)! "
                      "Logarithm of negative number is undefined.",
                      linear, val);
        } else if (val == 0.0f) {
            LOG_WARN("log_op WARNING: Input value at linear index %d is zero. "
                     "Logarithm of zero is -INF.",
                     linear);
        }
    }

    float* data = alloc_tensor_data(size, "log_op");
    if (!data)
        return;

    if (!can_use_simd_unary(in, out)) {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear) {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = logf(in->data->data[offset_in]);
        }
    } else {
        int i = 0;

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 z = Sleef_logf8_u10avx2(x);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            data[i] = logf(in->data->data[i]);
        }
    }
    from_data(out, data);
    free(data);
}

void exp_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: exp_op: Performing exponential");

    if (!check_tensors_unary(in, out, "exp_op"))
        return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "exp_op");
    if (!data)
        return;

    if (!can_use_simd_unary(in, out)) {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear) {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = expf(in->data->data[offset_in]);
        }
    } else {
        int i = 0;

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 z = Sleef_expf8_u10avx2(x);
            _mm256_storeu_ps(data + i, z);
        }

        for (; i < size; ++i) {
            data[i] = expf(in->data->data[i]);
        }
    }
    from_data(out, data);
    free(data);
}

void neg_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: neg_op: Performing negation");

    if (!check_tensors_unary(in, out, "neg_op"))
        return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "neg_op");
    if (!data)
        return;

    if (!can_use_simd_unary(in, out)) {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear) {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = 0.0f - in->data->data[offset_in];
        }
    } else {
        int i = 0;

        __m256 zeros = _mm256_setzero_ps();

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 y = _mm256_sub_ps(zeros, x);
            _mm256_storeu_ps(data + i, y);
        }

        for (; i < size; ++i) {
            data[i] = 0.0f - in->data->data[i];
        }
    }
    from_data(out, data);
    free(data);
}

void clip_op(Tensor* in, Tensor* out, float min_val, float max_val) {
    LOG_INFO("OP: clip_op: Performing clipping");

    if (!check_tensors_unary(in, out, "clip_op"))
        return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "clip_op");
    if (!data)
        return;

    if (!can_use_simd_unary(in, out)) {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear) {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            float val = in->data->data[offset_in];
            if (val < min_val) {
                data[offset_out] = min_val;
            } else if (val > max_val) {
                data[offset_out] = max_val;
            } else {
                data[offset_out] = val;
            }
        }
    } else {
        int i = 0;
        __m256 v_min = _mm256_set1_ps(min_val);
        __m256 v_max = _mm256_set1_ps(max_val);

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 y = _mm256_max_ps(x, v_min);
            y = _mm256_min_ps(y, v_max);
            _mm256_storeu_ps(data + i, y);
        }

        for (; i < size; ++i) {
            float val = in->data->data[i];
            if (val < min_val) {
                data[i] = min_val;
            } else if (val > max_val) {
                data[i] = max_val;
            } else {
                data[i] = val;
            }
        }
    }
    from_data(out, data);
    free(data);
}

void abs_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: abs_op: Performing absolute value");

    if (!check_tensors_unary(in, out, "abs_op"))
        return;

    int size = numel(in->shape, in->ndim);
    float* data = alloc_tensor_data(size, "abs_op");
    if (!data)
        return;

    if (!can_use_simd_unary(in, out)) {
        int offset_in, offset_out;
        for (int linear = 0; linear < size; ++linear) {
            COMPUTE_UNARY_OFFSETS(linear, in, offset_in, offset_out);
            data[offset_out] = in->data->data[offset_in] >= 0.0f ? in->data->data[offset_in] : 0.0f - in->data->data[offset_in];
        }
    } else {
        int i = 0;
        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // mask to remove sign bit

        for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
            __m256 x = _mm256_loadu_ps(in->data->data + i);
            __m256 y = _mm256_and_ps(x, mask);
            _mm256_storeu_ps(data + i, y);
        }

        for (; i < size; ++i) {
            data[i] = in->data->data[i] >= 0 ? in->data->data[i] : 0.0f - in->data->data[i];
        }
    }
    from_data(out, data);
    free(data);
}

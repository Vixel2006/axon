#include "logger.h"
#include "ops/ops.h"
#include "utils.h"
#include <immintrin.h>
#include <math.h>
#include <sleef.h>
#include <string.h>

#if DEBUG
#include "utils.h"
#endif

void relu_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: relu_op: Performing ReLU activation");

    if (!out->data->elems) {
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

            ((float*)out->data->elems)[offset_out] = ((float*)in->data->elems)[offset_in] > 0 ? ((float*)in->data->elems)[offset_in] : 0.0f;
        }
    } else {
        int i = 0;
        __m256 zeros = _mm256_setzero_ps();

        for (; i + 7 < size; i += 8) {
            __m256 vin = _mm256_loadu_ps(((float*)in->data->elems) + i);
            __m256 vout = _mm256_max_ps(vin, zeros);
            _mm256_storeu_ps(((float*)out->data->elems) + i, vout);
        }

        for (; i < size; ++i) {
            ((float*)out->data->elems)[i] = ((float*)in->data->elems)[i] > 0.0f ? ((float*)in->data->elems)[i] : 0.0f;
        }
    }
}

void log_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: log_op: Performing natural logarithm");

    if (!in || !out) {
        LOG_ERROR("log_op ERROR: Input or output tensor is NULL! in=%p, out=%p", (void*)in, (void*)out);
        return;
    }

    if (!out->data->elems) {
        LOG_ERROR("log_op ERROR: Output tensor data pointer is NULL after "
                  "reconfiguration.");
        return;
    }

    int size = numel(in->shape, in->ndim);

    for (int i = 0; i < size; ++i) {
        if (((float*)in->data->elems)[i] < 0.0f) {
            LOG_ERROR("log_op ERROR: Input value at index %d is negative (%.4f)! "
                      "Logarithm of negative number is undefined.",
                      i, ((float*)in->data->elems)[i]);
        } else if (((float*)in->data->elems)[i] == 0.0f) {
            LOG_WARN("log_op WARNING: Input value at index %d is zero. "
                     "Logarithm of zero is -INF.",
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

            ((float*)out->data->elems)[offset_out] = logf(((float*)in->data->elems)[offset_in]);
        }
    } else {
        int i = 0;

        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(((float*)in->data->elems) + i);
            __m256 z = Sleef_logf8_u10avx2(x);
            _mm256_storeu_ps(((float*)out->data->elems) + i, z);
        }

        for (; i < size; ++i) {
            ((float*)out->data->elems)[i] = logf(((float*)in->data->elems)[i]);
        }
    }
}

void exp_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: exp_op: Performing exponential");

    // Error checking for null tensors
    if (!in || !out) {
        LOG_ERROR("exp_op ERROR: Input or output tensor is NULL! in=%p, out=%p", (void*)in, (void*)out);
        return;
    }

    if (!out->data->elems) {
        LOG_ERROR("exp_op ERROR: Output tensor data pointer is NULL after "
                  "reconfiguration.");
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

            ((float*)out->data->elems)[offset_out] = expf(((float*)in->data->elems)[offset_in]);
        }
    } else {
        int i = 0;

        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(((float*)in->data->elems) + i);
            __m256 z = Sleef_expf8_u10avx2(x);
            _mm256_storeu_ps(((float*)out->data->elems) + i, z);
        }

        for (; i < size; ++i) {
            ((float*)out->data->elems)[i] = expf(((float*)in->data->elems)[i]);
        }
    }
}

void softmax_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: softmax_op: Performing softmax activation");
}

void neg_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: neg_op: Performing negation");

    // Error checking for null tensors
    if (!in || !out) {
        LOG_ERROR("neg_op ERROR: Input or output tensor is NULL! in=%p, out=%p", (void*)in, (void*)out);
        return;
    }

    if (!out->data->elems) {
        LOG_ERROR("neg_op ERROR: Output tensor data pointer is NULL after "
                  "reconfiguration.");
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

            ((float*)out->data->elems)[offset_out] = 0.0f - ((float*)in->data->elems)[offset_in];
        }

    } else {
        int i = 0;

        __m256 zeros = _mm256_setzero_ps();

        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(((float*)in->data->elems) + i);
            __m256 y = _mm256_sub_ps(zeros, x);
            _mm256_storeu_ps(((float*)out->data->elems) + i, y);
        }

        for (; i < size; ++i) {
            ((float*)out->data->elems)[i] = 0.0f - ((float*)in->data->elems)[i];
        }
    }
}

void clip_op(Tensor* in, float min_val, float max_val, Tensor* out) {
    LOG_INFO("OP: clip_op: Performing clipping");

    if (!in || !out) {
        LOG_ERROR("clip_op ERROR: Input or output tensor is NULL! in=%p, out=%p", (void*)in, (void*)out);
        return;
    }

    if (!out->data->elems) {
        LOG_ERROR("clip_op ERROR: Output tensor data pointer is NULL after "
                  "reconfiguration.");
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

            float val = ((float*)in->data->elems)[offset_in];
            if (val < min_val) {
                ((float*)out->data->elems)[offset_out] = min_val;
            } else if (val > max_val) {
                ((float*)out->data->elems)[offset_out] = max_val;
            } else {
                ((float*)out->data->elems)[offset_out] = val;
            }
        }
    } else {
        int i = 0;
        __m256 v_min = _mm256_set1_ps(min_val);
        __m256 v_max = _mm256_set1_ps(max_val);

        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(((float*)in->data->elems) + i);
            __m256 y = _mm256_max_ps(x, v_min);
            y = _mm256_min_ps(y, v_max);
            _mm256_storeu_ps(((float*)out->data->elems) + i, y);
        }

        for (; i < size; ++i) {
            float val = ((float*)in->data->elems)[i];
            if (val < min_val) {
                ((float*)out->data->elems)[i] = min_val;
            } else if (val > max_val) {
                ((float*)out->data->elems)[i] = max_val;
            } else {
                ((float*)out->data->elems)[i] = val;
            }
        }
    }
}

void abs_op(Tensor* in, Tensor* out) {
    LOG_INFO("OP: abs_op: Performing absolute value");

    if (!out->data->elems) {
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

            ((float*)out->data->elems)[offset_out] = ((float*)in->data->elems)[offset_in] >= 0.0f ? ((float*)in->data->elems)[offset_in] : 0.0f - ((float*)in->data->elems)[offset_in];
        }
    } else {
        int i = 0;
        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // mask to remove sign bit

        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(((float*)in->data->elems) + i);
            __m256 y = _mm256_and_ps(x, mask);
            _mm256_storeu_ps(((float*)out->data->elems) + i, y);
        }

        for (; i < size; ++i) {
            ((float*)out->data->elems)[i] = ((float*)in->data->elems)[i] >= 0 ? ((float*)in->data->elems)[i] : 0.0f - ((float*)in->data->elems)[i];
        }
    }
}

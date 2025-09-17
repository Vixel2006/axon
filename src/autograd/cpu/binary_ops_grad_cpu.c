#include "autograd/autograd_binary.h"

#define SIMD_WIDTH 8

typedef float (*unary_grad_fn)(float dout, float aval, float scalar);
typedef float (*binary_grad_fn)(float dout, float aval, float bval);

static inline float binary_add_da(float dout, float a, float b) {
    return dout;
}
static inline float binary_add_db(float dout, float a, float b) {
    return dout;
}
static inline float binary_sub_da(float dout, float a, float b) {
    return dout;
}
static inline float binary_sub_db(float dout, float a, float b) {
    return -dout;
}
static inline float binary_mul_da(float dout, float a, float b) {
    return dout * b;
}
static inline float binary_mul_db(float dout, float a, float b) {
    return dout * a;
}
static inline float binary_div_da(float dout, float a, float b) {
    return (b != 0.0f) ? dout / b : 0.0f;
}
static inline float binary_div_db(float dout, float a, float b) {
    return (b != 0.0f) ? -dout * a / (b * b) : 0.0f;
}

static inline float unary_add_da(float dout, float a, float scalar) {
    return dout;
}
static inline float unary_sub_da(float dout, float a, float scalar) {
    return dout;
}
static inline float unary_rsub_da(float dout, float a, float scalar) {
    return -dout;
}
static inline float unary_mul_da(float dout, float a, float scalar) {
    return dout * scalar;
}
static inline float unary_div_da(float dout, float a, float scalar) {
    return (scalar != 0.0f) ? dout / scalar : 0.0f;
}
static inline float unary_rdiv_da(float dout, float a, float scalar) {
    return (a != 0.0f) ? -scalar / (a * a) * dout : 0.0f;
}
static inline float unary_pow_da(float dout, float a, float scalar) {
    float grad_val = 0.0f;
    if (!(a == 0.0f && (scalar - 1.0f) < 0.0f)) {
        grad_val = scalar * powf(a, scalar - 1.0f);
    }
    return dout * grad_val;
}

void unary_grad_noncontig(Tensor* out, Tensor* a, float scalar, unary_grad_fn da_fn) {
    if (!a->requires_grad)
        return;
    int size = numel(out->shape, out->ndim);
    int ndim = out->ndim;
    int* shape = out->shape;
    int* a_strides = a->strides;
    int* out_strides = out->strides;
    float* a_grad = a->grad->data->data;
    float* a_data = a->data->data;
    float* out_grad = out->grad->data->data;
    for (int linear = 0; linear < size; ++linear) {
        int idx = linear;
        int a_offset = 0, out_offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            int coord = idx % shape[d];
            idx /= shape[d];
            a_offset += coord * a_strides[d];
            out_offset += coord * out_strides[d];
        }
        float dout = out_grad[out_offset];
        float aval = a_data[a_offset];
        a_grad[a_offset] += da_fn(dout, aval, scalar);
    }
}

void binary_grad_noncontig(Tensor* out, Tensor* a, Tensor* b, binary_grad_fn da_fn, binary_grad_fn db_fn) {
    int size = numel(out->shape, out->ndim);
    int ndim = out->ndim;
    int* shape = out->shape;
    int* a_strides = a->strides;
    int* b_strides = b->strides;
    int* out_strides = out->strides;
    float* a_data = a->data->data;
    float* b_data = b->data->data;
    float* out_grad = out->grad->data->data;
    float* a_grad = (a->requires_grad ? a->grad->data->data : NULL);
    float* b_grad = (b->requires_grad ? b->grad->data->data : NULL);
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
        float dout = out_grad[out_offset];
        float aval = a_data[a_offset];
        float bval = b_data[b_offset];
        if (a_grad)
            a_grad[a_offset] += da_fn(dout, aval, bval);
        if (b_grad)
            b_grad[b_offset] += db_fn(dout, aval, bval);
    }
}

void add_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: add_grad_op: Computing gradient for addition");

    // Error checking for null tensors
    if (!out || !out->grad->data || !prev) {
        LOG_ERROR("add_grad_op ERROR: Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*)out, (void*)out->grad->data, (void*)prev);
        return;
    }

    if (n_prev != 1 && n_prev != 2) {
        LOG_ERROR("add_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 1 or 2.",
                  n_prev);
        return;
    }

    int size = numel(out->shape, out->ndim);

    if (n_prev == 2) {
        Tensor* a = prev[0];
        Tensor* b = prev[1];

        if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
            binary_grad_noncontig(out, a, b, binary_add_da, binary_add_db);
        } else {
            if (a->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_add_ps(a_grad, dout);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i) {
                    a->grad->data->data[i] += out->grad->data->data[i];
                }
            }

            if (b->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 b_grad = _mm256_loadu_ps(b->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 db = _mm256_add_ps(b_grad, dout);
                    _mm256_storeu_ps(b->grad->data->data + i, db);
                }

                for (; i < size; ++i) {
                    b->grad->data->data[i] += out->grad->data->data[i];
                }
            }
        }
    } else if (n_prev == 1 && extras != NULL) {
        Tensor* a = prev[0];
        float b = *((float*)extras);

        if (!is_contiguous(a) || !is_contiguous(out)) {
            unary_grad_noncontig(out, a, b, unary_add_da);
        } else {
            if (a->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_add_ps(a_grad, dout);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i) {
                    a->grad->data->data[i] += out->grad->data->data[i];
                }
            }
        }
    }
}

void sub_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: sub_grad_op: Computing gradient for subtraction");

    int size = numel(out->shape, out->ndim);

    if (n_prev == 2) {
        Tensor* a = prev[0];
        Tensor* b = prev[1];

        if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
            binary_grad_noncontig(out, a, b, binary_sub_da, binary_sub_db);
        } else {
            if (a->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_add_ps(a_grad, dout);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i) {
                    a->grad->data->data[i] += out->grad->data->data[i];
                }
            }

            if (b->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 b_grad = _mm256_loadu_ps(b->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 db = _mm256_sub_ps(b_grad, dout);
                    _mm256_storeu_ps(b->grad->data->data + i, db);
                }

                for (; i < size; ++i) {
                    b->grad->data->data[i] -= out->grad->data->data[i];
                }
            }
        }
    } else if (n_prev == 1 && extras != NULL) {
        Tensor* a = prev[0];
        float b = *((float*)extras);

        if (!is_contiguous(a) || !is_contiguous(out)) {
            unary_grad_noncontig(out, a, b, unary_sub_da);
        } else {
            if (a->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_add_ps(a_grad, dout);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i) {
                    a->grad->data->data[i] += out->grad->data->data[i];
                }
            }
        }
    }
}

void rsub_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: rsub_grad_op: Computing gradient for reverse "
             "subtraction");

    // Error checking for null tensors and invalid n_prev
    if (!out || !out->grad->data || !prev) {
        LOG_ERROR("rsub_grad_op ERROR: Output tensor, output gradient, or previous "
                  "tensors array is NULL! out=%p, out->grad=%p, prev=%p",
                  (void*)out, (void*)out->grad->data, (void*)prev);
        return;
    }

    if (n_prev != 1) {
        LOG_ERROR("rsub_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 1.",
                  n_prev);
        return;
    }

    if (!prev[0]) {
        LOG_ERROR("rsub_grad_op ERROR: Previous tensor is NULL! prev[0]=%p", (void*)prev[0]);
        return;
    }

    if (!extras) {
        LOG_ERROR("rsub_grad_op ERROR: Extras is NULL (scalar value missing)!");
        return;
    }

    Tensor* a = prev[0];
    float b = *((float*)extras);

    int size = numel(out->shape, out->ndim);

    if (a->requires_grad) {
        if (!is_contiguous(a) || !is_contiguous(out)) {
            unary_grad_noncontig(out, a, b, unary_rsub_da);
        } else {
            int i = 0;
            for (; i + 7 < size; i += 8) {
                __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                __m256 da = _mm256_sub_ps(a_grad, dout);
                _mm256_storeu_ps(a->grad->data->data + i, da);
            }

            for (; i < size; ++i) {
                a->grad->data->data[i] -= out->grad->data->data[i];
            }
        }
    }
}

void mul_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: mul_grad_op: Computing gradient for multiplication");

    int size = numel(out->shape, out->ndim);

    if (n_prev == 2) {
        Tensor* a = prev[0];
        Tensor* b = prev[1];

        if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
            binary_grad_noncontig(out, a, b, binary_mul_da, binary_mul_db);
        } else {
            if (a->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_fmadd_ps(b_data, dout, a_grad);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i) {
                    a->grad->data->data[i] += out->grad->data->data[i] * b->data->data[i];
                }
            }

            if (b->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 b_grad = _mm256_loadu_ps(b->grad->data->data + i);
                    __m256 a_data = _mm256_loadu_ps(a->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 db = _mm256_fmadd_ps(a_data, dout, b_grad);
                    _mm256_storeu_ps(b->grad->data->data + i, db);
                }

                for (; i < size; ++i) {
                    b->grad->data->data[i] += out->grad->data->data[i] * a->data->data[i];
                }
            }
        }
    } else if (n_prev == 1 && extras != NULL) {
        Tensor* a = prev[0];
        float b = *((float*)extras);

        if (!is_contiguous(a) || !is_contiguous(out)) {
            unary_grad_noncontig(out, a, b, unary_mul_da);
        } else {
            if (a->requires_grad) {
                int i = 0;
                __m256 scalar = _mm256_set1_ps(b);
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_fmadd_ps(scalar, dout, a_grad);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i) {
                    a->grad->data->data[i] += out->grad->data->data[i] * b;
                }
            }
        }
    }
}

void pow_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: pow_grad_op: Computing gradient for power operation");

    int size = numel(out->shape, out->ndim);

    Tensor* a = prev[0];
    float b = *((float*)extras);

    if (!a->requires_grad)
        return;

    if (!is_contiguous(a) || !is_contiguous(out)) {
        unary_grad_noncontig(out, a, b, unary_pow_da);
    } else {
        int i = 0;
        __m256 scalar_b = _mm256_set1_ps(b);
        float c = b - 1.0f;
        __m256 scalar_bm1 = _mm256_set1_ps(c);
        __m256 zero = _mm256_setzero_ps();

        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(a->data->data + i);
            __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
            __m256 agrad = _mm256_loadu_ps(a->grad->data->data + i);

            __m256 x_pow = Sleef_powf8_u10avx2(x, scalar_bm1);
            __m256 coeff = _mm256_mul_ps(scalar_b, x_pow);

            // mask problematic case: x==0 && (b-1)<0
            __m256 zero_mask = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
            __m256 neg_exp_mask = _mm256_cmp_ps(scalar_bm1, zero, _CMP_LT_OQ);
            __m256 problem_mask = _mm256_and_ps(zero_mask, neg_exp_mask);
            coeff = _mm256_blendv_ps(coeff, zero, problem_mask);

            __m256 da = _mm256_fmadd_ps(dout, coeff, agrad);
            _mm256_storeu_ps(a->grad->data->data + i, da);
        }

        for (; i < size; ++i) {
            float x = a->data->data[i];
            float grad_val = 0.0f;

            if (!(x == 0.0f && (b - 1.0f) < 0.0f)) {
                grad_val = b * powf(x, b - 1.0f);
            }
            a->grad->data->data[i] += out->grad->data->data[i] * grad_val;
        }
    }
}

void div_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: div_grad_op: Computing gradient for division");

    // Basic null pointer checks
    if (!out || !prev) {
        LOG_ERROR("div_grad_op ERROR: Output tensor or previous tensors array "
                  "is NULL! out=%p, prev=%p",
                  (void*)out, (void*)prev);
        return;
    }

    if (!out->grad->data->data) {
        LOG_ERROR("div_grad_op ERROR: Output gradient is NULL! out->grad=%p", (void*)out->grad->data->data);
        return;
    }

    if (n_prev != 2 && n_prev != 1) {
        LOG_ERROR("div_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 1 or 2.",
                  n_prev);
        return;
    }

    if (n_prev == 2) {
        if (!prev[0] || !prev[1]) {
            LOG_ERROR("div_grad_op ERROR: One or both previous tensors are NULL! "
                      "prev[0]=%p, prev[1]=%p",
                      (void*)prev[0], (void*)prev[1]);
            return;
        }
        if (!prev[0]->data->data || !prev[1]->data->data) {
            LOG_ERROR("div_grad_op ERROR: One or both previous tensors' data are "
                      "NULL!");
            return;
        }
        if (prev[0]->requires_grad && prev[0]->grad->data == NULL) {
            LOG_ERROR("div_grad_op ERROR: Previous tensor 0 requires grad but its "
                      "grad is NULL!");
            return;
        }
        if (prev[1]->requires_grad && prev[1]->grad->data == NULL) {
            LOG_ERROR("div_grad_op ERROR: Previous tensor 1 requires grad but its "
                      "grad is NULL!");
            return;
        }
    }

    int size = numel(out->shape, out->ndim);

    if (n_prev == 2) {
        Tensor* a = prev[0];
        Tensor* b = prev[1];

        if (!is_contiguous(a) || !is_contiguous(b) || !is_contiguous(out)) {
            binary_grad_noncontig(out, a, b, binary_div_da, binary_div_db);
        } else {
            __m256 zero = _mm256_setzero_ps();
            if (a->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                    __m256 mask = _mm256_cmp_ps(b_data, zero, _CMP_EQ_OQ);
                    __m256 div = _mm256_div_ps(dout, b_data);
                    div = _mm256_blendv_ps(div, zero, mask);
                    __m256 da = _mm256_add_ps(a_grad, div);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }
                for (; i < size; ++i) {
                    float bb = b->data->data[i];
                    a->grad->data->data[i] += (bb != 0.0f ? out->grad->data->data[i] / bb : 0.0f);
                }
            }

            if (b->requires_grad) {
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 b_grad = _mm256_loadu_ps(b->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 a_data = _mm256_loadu_ps(a->data->data + i);
                    __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                    __m256 b_sq = _mm256_mul_ps(b_data, b_data);
                    __m256 mask = _mm256_cmp_ps(b_sq, zero, _CMP_EQ_OQ);
                    __m256 term = _mm256_mul_ps(dout, a_data);
                    __m256 div = _mm256_div_ps(term, b_sq);
                    div = _mm256_blendv_ps(div, zero, mask);
                    __m256 db = _mm256_sub_ps(b_grad, div);
                    _mm256_storeu_ps(b->grad->data->data + i, db);
                }
                for (; i < size; ++i) {
                    float bb = b->data->data[i];
                    if (bb != 0.0f) {
                        b->grad->data->data[i] -= out->grad->data->data[i] * a->data->data[i] / (bb * bb);
                    }
                }
            }
        }
    } else if (n_prev == 1 && extras != NULL) {
        Tensor* a = prev[0];
        float b = *((float*)extras);

        if (!is_contiguous(a) || !is_contiguous(out)) {
            unary_grad_noncontig(out, a, b, unary_div_da);
        } else {
            if (a->requires_grad) {
                float inv_b = (b != 0.0f ? 1.0f / b : 0.0f);
                __m256 inv = _mm256_set1_ps(inv_b);
                int i = 0;
                for (; i + 7 < size; i += 8) {
                    __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                    __m256 da = _mm256_fmadd_ps(inv, dout, a_grad);
                    _mm256_storeu_ps(a->grad->data->data + i, da);
                }

                for (; i < size; ++i) {
                    a->grad->data->data[i] += (b != 0.0f ? out->grad->data->data[i] / b : 0.0f);
                }
            }
        }
    }
}

void rdiv_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: rdiv_grad_op: Computing gradient for reverse "
             "division");

    Tensor* a = prev[0];
    float b = *((float*)extras);

    int size = numel(out->shape, out->ndim);

    if (a->requires_grad) {
        if (!is_contiguous(a) || !is_contiguous(out)) {
            unary_grad_noncontig(out, a, b, unary_rdiv_da);
        } else {
            int i = 0;
            __m256 neg_b = _mm256_set1_ps(-b);
            for (; i + 7 < size; i += 8) {
                __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                __m256 a_data = _mm256_loadu_ps(a->data->data + i);
                __m256 dout = _mm256_loadu_ps(out->grad->data->data + i);
                __m256 a_squared = _mm256_mul_ps(a_data, a_data);
                __m256 da = _mm256_fmadd_ps(_mm256_div_ps(neg_b, a_squared), dout, a_grad);
                _mm256_storeu_ps(a->grad->data->data + i, da);
            }

            for (; i < size; ++i) {
                float aa = a->data->data[i];
                a->grad->data->data[i] += (aa != 0.0f ? out->grad->data->data[i] * (-b) / (aa * aa) : 0.0f);
            }
        }
    }
}

void matmul_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: matmul_grad_op: Computing gradient for matrix "
             "multiplication");
    if (!out || !prev) {
        LOG_ERROR("matmul_grad_op ERROR: Output tensor or previous tensors array "
                  "is NULL! out=%p, prev=%p",
                  (void*)out, (void*)prev);
        return;
    }

    if (!out->grad->data->data) {
        LOG_ERROR("matmul_grad_op ERROR: Output gradient is NULL! out->grad=%p", (void*)out->grad->data->data);
        return;
    }

    if (n_prev != 2) {
        LOG_ERROR("matmul_grad_op ERROR: Invalid number of previous tensors: %d. "
                  "Expected 2.",
                  n_prev);
        return;
    }

    if (!prev[0] || !prev[1]) {
        LOG_ERROR("matmul_grad_op ERROR: One or both previous tensors are NULL! "
                  "prev[0]=%p, prev[1]=%p",
                  (void*)prev[0], (void*)prev[1]);
        return;
    }

    Tensor* a = prev[0];
    Tensor* b = prev[1];

    if (a->ndim < 2 || b->ndim < 2 || out->ndim < 2) {
        LOG_ERROR("matmul_grad_op ERROR: All tensors must have at least 2 "
                  "dimensions! a->ndim=%d, b->ndim=%d, out->ndim=%d",
                  a->ndim, b->ndim, out->ndim);
        return;
    }

    if (!a->shape || !b->shape || !out->shape) {
        LOG_ERROR("matmul_grad_op ERROR: One or more shape arrays are NULL!");
        return;
    }

    if (!a->strides || !b->strides || !out->strides) {
        LOG_ERROR("matmul_grad_op ERROR: One or more stride arrays are NULL!");
        return;
    }

    if (!a->data->data || !b->data->data) {
        LOG_ERROR("matmul_grad_op ERROR: One or more data arrays are NULL!");
        return;
    }

    int N = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int M = b->shape[b->ndim - 1];

    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) {
        LOG_ERROR("matmul_grad_op ERROR: Dimension mismatch for matrix multiplication! "
                  "a->shape[last]=%d, b->shape[second_last]=%d",
                  a->shape[a->ndim - 1], b->shape[b->ndim - 2]);
        return;
    }

    // Validate output dimensions match expected result
    if (out->shape[out->ndim - 2] != N || out->shape[out->ndim - 1] != M) {
        LOG_ERROR("matmul_grad_op ERROR: Output dimensions don't match expected result! "
                  "Expected (%d, %d), got (%d, %d)",
                  N, M, out->shape[out->ndim - 2], out->shape[out->ndim - 1]);
        return;
    }

    // Calculate total batch size (product of all batch dimensions)
    int batch_size = 1;
    for (int i = 0; i < out->ndim - 2; ++i) {
        int a_dim = (i < a->ndim - 2) ? a->shape[i] : 1;
        int b_dim = (i < b->ndim - 2) ? b->shape[i] : 1;
        int out_dim = (i < out->ndim - 2) ? out->shape[i] : 1;

        // Verify broadcasting compatibility
        if ((a_dim != 1 && b_dim != 1 && a_dim != b_dim) || (a_dim != 1 && out_dim != 1 && a_dim != out_dim) || (b_dim != 1 && out_dim != 1 && b_dim != out_dim)) {
            LOG_ERROR("matmul_grad_op ERROR: Incompatible batch dimensions at index %d: "
                      "a=%d, b=%d, out=%d",
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

    // Compute gradient for tensor a: grad_a += out_grad @ b^T
    if (a->requires_grad) {
        if (!a->grad->data->data) {
            LOG_ERROR("matmul_grad_op ERROR: Tensor 'a' requires grad but its grad "
                      "is NULL!");
            return;
        }

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            // Calculate batch offsets with proper broadcasting
            int a_batch_offset = 0;
            int b_batch_offset = 0;
            int out_batch_offset = 0;

            int temp_batch_idx = batch_idx;
            for (int dim = out->ndim - 3; dim >= 0; --dim) {
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
                        float out_grad_val = out->grad->data->data[out_batch_offset + i * out_row_stride + k * out_col_stride];
                        float b_val = b->data->data[b_batch_offset + j * b_row_stride + k * b_col_stride];
                        sum += out_grad_val * b_val;
                    }
                    a->grad->data->data[a_batch_offset + i * a_row_stride + j * a_col_stride] += sum;
                }
            }
        }
    }

    // Compute gradient for tensor b: grad_b += a^T @ out_grad
    if (b->requires_grad) {
        if (!b->grad->data->data) {
            LOG_ERROR("matmul_grad_op ERROR: Tensor 'b' requires grad but its grad "
                      "is NULL!");
            return;
        }

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            // Calculate batch offsets with proper broadcasting
            int a_batch_offset = 0;
            int b_batch_offset = 0;
            int out_batch_offset = 0;

            int temp_batch_idx = batch_idx;
            for (int dim = out->ndim - 3; dim >= 0; --dim) {
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
                        float a_val = a->data->data[a_batch_offset + k * a_row_stride + i * a_col_stride];
                        float out_grad_val = out->grad->data->data[out_batch_offset + k * out_row_stride + j * out_col_stride];
                        sum += a_val * out_grad_val;
                    }
                    b->grad->data->data[b_batch_offset + i * b_row_stride + j * b_col_stride] += sum;
                }
            }
        }
    }

    LOG_INFO("GRAD: matmul_grad_op: Gradient computation completed "
             "successfully");
}

void conv2d_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: conv2d_grad_op: Computing gradient for 2D convolution");

    Tensor* in = prev[0];
    Tensor* kernel = prev[1];

    BackwardConvExtras* conv_extras = (BackwardConvExtras*)extras;

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
                                            float out_grad_val = out->grad->data->data[n * Cout * Hout * Wout + cout * Hout * Wout + oh * Wout + ow];

                                            for (int cin = 0; cin < Cin; ++cin) {
                                                int in_idx = n * Cin * Hin * Win + cin * Hin * Win + ih * Win + iw;
                                                int kernel_grad_idx = cout * Cin * Kh * Kw + cin * Kh * Kw + kh * Kw + kw;

                                                kernel->grad->data->data[kernel_grad_idx] += in->data->data[in_idx] * out_grad_val;
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
                            int oh_end = (oh_start + TILE_H > Hout) ? Hout : oh_start + TILE_H;

                            for (int ow_start = 0; ow_start < Wout; ow_start += TILE_W) {
                                int ow_end = (ow_start + TILE_W > Wout) ? Wout : ow_start + TILE_W;

                                for (int oh = oh_start; oh < oh_end; ++oh) {
                                    for (int ow = ow_start; ow < ow_end; ++ow) {
                                        int ih = oh * Sh - padding + kh;
                                        int iw = ow * Sw - padding + kw;

                                        if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                                            float out_grad_val = out->grad->data->data[n * Cout * Hout * Wout + cout * Hout * Wout + oh * Wout + ow];

                                            for (int cin = 0; cin < Cin; ++cin) {
                                                int kernel_idx = cout * Cin * Kh * Kw + cin * Kh * Kw + kh * Kw + kw;
                                                int in_grad_idx = n * Cin * Hin * Win + cin * Hin * Win + ih * Win + iw;

                                                in->grad->data->data[in_grad_idx] += kernel->data->data[kernel_idx] * out_grad_val;
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

void dot_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: dot_grad_op: Computing gradient for dot product");

    Tensor* a = prev[0];
    Tensor* b = prev[1];

    int size = numel(a->shape, a->ndim);
    float dout = out->grad->data->data[0];

    if (!is_contiguous(a) || !is_contiguous(b)) {
        if (a->requires_grad) {
            for (int linear = 0; linear < size; ++linear) {
                int idx = linear;
                int a_offset = 0, b_offset = 0;

                for (int d = a->ndim - 1; d >= 0; --d) {
                    int coord = idx % a->shape[d];
                    idx /= a->shape[d];

                    a_offset += coord * a->strides[d];
                    b_offset += coord * b->strides[d];
                }
                a->grad->data->data[a_offset] += dout * b->data->data[b_offset];
            }
        }

        if (b->requires_grad) {
            for (int linear = 0; linear < size; ++linear) {
                int idx = linear;
                int a_offset = 0, b_offset = 0;

                for (int d = a->ndim - 1; d >= 0; --d) {
                    int coord = idx % a->shape[d];
                    idx /= a->shape[d];

                    a_offset += coord * a->strides[d];
                    b_offset += coord * b->strides[d];
                }
                b->grad->data->data[b_offset] += dout * a->data->data[a_offset];
            }
        }
    } else {
        if (a->requires_grad) {
            int i = 0;
            __m256 dout_vec = _mm256_set1_ps(dout);
            for (; i + 7 < size; i += 8) {
                __m256 a_grad = _mm256_loadu_ps(a->grad->data->data + i);
                __m256 b_data = _mm256_loadu_ps(b->data->data + i);
                __m256 da = _mm256_fmadd_ps(dout_vec, b_data, a_grad);
                _mm256_storeu_ps(a->grad->data->data + i, da);
            }

            for (; i < size; ++i) {
                a->grad->data->data[i] += dout * b->data->data[i];
            }
        }

        if (b->requires_grad) {
            int i = 0;
            __m256 dout_vec = _mm256_set1_ps(dout);
            for (; i + 7 < size; i += 8) {
                __m256 b_grad = _mm256_loadu_ps(b->grad->data->data + i);
                __m256 a_data = _mm256_loadu_ps(a->data->data + i);
                __m256 db = _mm256_fmadd_ps(dout_vec, a_data, b_grad);
                _mm256_storeu_ps(b->grad->data->data + i, db);
            }

            for (; i < size; ++i) {
                b->grad->data->data[i] += dout * a->data->data[i];
            }
        }
    }
}

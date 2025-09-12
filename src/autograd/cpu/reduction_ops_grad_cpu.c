#include "utils.h"
#include <immintrin.h>
#include <stdlib.h>

#include "autograd/autograd.h"
#include "autograd/autograd_utils.h"
#include "logger.h"

#define SIMD_WIDTH 8

void max_full_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras);

void sum_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: sum_grad_op: Computing gradient for sum reduction");

    Tensor* in = prev[0];

    if (!in->requires_grad) {
        return;
    }

    int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

    if (reduced_dim == -1) {
        int size = numel(in->shape, in->ndim);
        if (!is_contiguous(in) || !is_contiguous(out)) {
            int* in_strides = in->strides;
            int* out_strides = out->strides;
            int* shape = in->shape;
            int ndim = in->ndim;

            for (int linear = 0; linear < size; ++linear) {
                int idx = linear;
                int in_offset = 0, out_offset = 0;

                for (int d = ndim - 1; d >= 0; --d) {
                    int coord = idx % shape[d];
                    idx /= shape[d];

                    in_offset += coord * in_strides[d];
                    out_offset += coord * out_strides[d];
                }
                ((float*)in->grad->elems)[in_offset] += ((float*)out->grad->elems)[out_offset];
            }
        } else {
            int i = 0;
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
                __m256 in_grad = _mm256_loadu_ps(((float*)in->grad->elems) + i);
                __m256 dout = _mm256_loadu_ps(((float*)out->grad->elems) + i);
                __m256 new_in_grad = _mm256_add_ps(in_grad, dout);
                _mm256_storeu_ps(((float*)in->grad->elems) + i, new_in_grad);
            }
            for (; i < size; ++i) {
                ((float*)in->grad->elems)[i] += ((float*)out->grad->elems)[i];
            }
        }
        return;
    }

    // Reduction case
    int in_size = numel(in->shape, in->ndim);
    int* in_strides = in->strides;
    int* in_shape = in->shape;
    int in_ndim = in->ndim;

    int* out_strides = out->strides;
    int* out_shape = out->shape;
    int out_ndim = out->ndim;

    if (!is_contiguous(in) || !is_contiguous(out)) {
        // Non-contiguous path
        for (int in_linear_idx = 0; in_linear_idx < in_size; ++in_linear_idx) {
            int temp_in_linear_idx = in_linear_idx;
            int in_offset = 0;
            int out_offset = 0;
            int current_out_dim_idx = 0;

            for (int d = in_ndim - 1; d >= 0; --d) {
                int coord = temp_in_linear_idx % in_shape[d];
                temp_in_linear_idx /= in_shape[d];
                in_offset += coord * in_strides[d];

                if (d != reduced_dim) {
                    out_offset += coord * out_strides[current_out_dim_idx];
                    current_out_dim_idx++;
                }
            }
            ((float*)in->grad->elems)[in_offset] += ((float*)out->grad->elems)[out_offset];
        }
    } else {
        // Contiguous path (existing code)
        int reduce_size = in->shape[reduced_dim];
        int num_batches = numel(out->shape, out->ndim);

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            float grad = ((float*)out->grad->elems)[batch_idx];

            __m256 grad_vec = _mm256_set1_ps(grad);

            int base_offset = batch_idx * reduce_size;

            int i = 0;
            for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH) {
                __m256 data_vec = _mm256_loadu_ps(((float*)in->grad->elems) + base_offset + i);
                data_vec = _mm256_add_ps(data_vec, grad_vec);
                _mm256_storeu_ps(((float*)in->grad->elems) + base_offset + i, data_vec);
            }

            for (; i < reduce_size; ++i) {
                ((float*)in->grad->elems)[base_offset + i] += grad;
            }
        }
    }
}

void mean_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: mean_grad_op: Computing gradient for mean reduction");

    Tensor* in = prev[0];

    if (!in->requires_grad) {
        return;
    }

    int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

    if (reduced_dim == -1) {
        int size = numel(in->shape, in->ndim);
        if (!is_contiguous(in) || !is_contiguous(out)) {
            int* in_strides = in->strides;
            int* out_strides = out->strides;
            int* shape = in->shape;
            int ndim = in->ndim;

            for (int linear = 0; linear < size; ++linear) {
                int idx = linear;
                int in_offset = 0, out_offset = 0;

                for (int d = ndim - 1; d >= 0; --d) {
                    int coord = idx % shape[d];
                    idx /= shape[d];

                    in_offset += coord * in_strides[d];
                    out_offset += coord * out_strides[d];
                }
                ((float*)in->grad->elems)[in_offset] += ((float*)out->grad->elems)[out_offset];
            }
        } else {
            int i = 0;
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
                __m256 in_grad = _mm256_loadu_ps(((float*)in->grad->elems) + i);
                __m256 dout = _mm256_loadu_ps(((float*)out->grad->elems) + i);
                __m256 new_in_grad = _mm256_add_ps(in_grad, dout);
                _mm256_storeu_ps(((float*)in->grad->elems) + i, new_in_grad);
            }
            for (; i < size; ++i) {
                ((float*)in->grad->elems)[i] += ((float*)out->grad->elems)[i];
            }
        }
        return;
    }

    // Reduction case
    int in_size = numel(in->shape, in->ndim);
    int* in_strides = in->strides;
    int* in_shape = in->shape;
    int in_ndim = in->ndim;

    int* out_strides = out->strides;
    int* out_shape = out->shape;
    int out_ndim = out->ndim;

    int reduce_size = in->shape[reduced_dim];

    if (!is_contiguous(in) || !is_contiguous(out)) {
        // Non-contiguous path
        for (int in_linear_idx = 0; in_linear_idx < in_size; ++in_linear_idx) {
            int temp_in_linear_idx = in_linear_idx;
            int in_offset = 0;
            int out_offset = 0;
            int current_out_dim_idx = 0;

            for (int d = in_ndim - 1; d >= 0; --d) {
                int coord = temp_in_linear_idx % in_shape[d];
                temp_in_linear_idx /= in_shape[d];
                in_offset += coord * in_strides[d];

                if (d != reduced_dim) {
                    out_offset += coord * out_strides[current_out_dim_idx];
                    current_out_dim_idx++;
                }
            }
            ((float*)in->grad->elems)[in_offset] += ((float*)out->grad->elems)[out_offset] / reduce_size;
        }
    } else {
        // Contiguous path (existing code)
        int num_batches = numel(out->shape, out->ndim);

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            float grad = ((float*)out->grad->elems)[batch_idx];

            __m256 grad_vec = _mm256_set1_ps(grad / reduce_size);

            int base_offset = batch_idx * reduce_size;

            int i = 0;
            for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH) {
                __m256 data_vec = _mm256_loadu_ps(((float*)in->grad->elems) + base_offset + i);
                data_vec = _mm256_add_ps(data_vec, grad_vec);
                _mm256_storeu_ps(((float*)in->grad->elems) + base_offset + i, data_vec);
            }

            for (; i < reduce_size; ++i) {
                ((float*)in->grad->elems)[base_offset + i] += grad / reduce_size;
            }
        }
    }
}

void max_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: max_grad_op: Computing gradient for max reduction");

    Tensor* in = prev[0];

    if (!in->requires_grad) {
        return;
    }

    // If output is a scalar, it's a full reduction. Delegate to max_full_grad_op.
    if (out->ndim == 0) {
        max_full_grad_op(out, prev, n_prev, extras);
        return;
    }

    int reduced_dim = get_reduced_dim(in->shape, out->shape, in->ndim, out->ndim);

    if (reduced_dim == -1) {
        // Handle case where no reduction happened (in and out have same shape)
        int size = numel(in->shape, in->ndim);
        if (!is_contiguous(in) || !is_contiguous(out)) {
            int* in_strides = in->strides;
            int* out_strides = out->strides;
            int* shape = in->shape;
            int ndim = in->ndim;

            for (int linear = 0; linear < size; ++linear) {
                int idx = linear;
                int in_offset = 0, out_offset = 0;

                for (int d = ndim - 1; d >= 0; --d) {
                    int coord = idx % shape[d];
                    idx /= shape[d];

                    in_offset += coord * in_strides[d];
                    out_offset += coord * out_strides[d];
                }
                ((float*)in->grad->elems)[in_offset] += ((float*)out->grad->elems)[out_offset];
            }
        } else {
            int i = 0;
            for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
                __m256 in_grad = _mm256_loadu_ps(((float*)in->grad->elems) + i);
                __m256 dout = _mm256_loadu_ps(((float*)out->grad->elems) + i);
                __m256 new_in_grad = _mm256_add_ps(in_grad, dout);
                _mm256_storeu_ps(((float*)in->grad->elems) + i, new_in_grad);
            }
            for (; i < size; ++i) {
                ((float*)in->grad->elems)[i] += ((float*)out->grad->elems)[i];
            }
        }
        return;
    }

    // Reduction case
    int in_size = numel(in->shape, in->ndim);
    int* in_strides = in->strides;
    int* in_shape = in->shape;
    int in_ndim = in->ndim;

    int* out_strides = out->strides;
    int* out_shape = out->shape;
    int out_ndim = out->ndim;

    if (!is_contiguous(in) || !is_contiguous(out)) {
        // Non-contiguous path
        for (int in_linear_idx = 0; in_linear_idx < in_size; ++in_linear_idx) {
            int temp_in_linear_idx = in_linear_idx;
            int in_offset = 0;
            int out_offset = 0;
            int current_out_dim_idx = 0;

            for (int d = in_ndim - 1; d >= 0; --d) {
                int coord = temp_in_linear_idx % in_shape[d];
                temp_in_linear_idx /= in_shape[d];
                in_offset += coord * in_strides[d];

                if (d != reduced_dim) {
                    out_offset += coord * out_strides[current_out_dim_idx];
                    current_out_dim_idx++;
                }
            }

            if (((float*)in->data->elems)[in_offset] == ((float*)out->data->elems)[out_offset]) {
                ((float*)in->grad->elems)[in_offset] += ((float*)out->grad->elems)[out_offset];
            }
        }
    } else {
        // Contiguous path (existing code with bug fix)
        int reduce_size = in->shape[reduced_dim];
        int num_batches = numel(out->shape, out->ndim);

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            float grad = ((float*)out->grad->elems)[batch_idx];
            float max = ((float*)out->data->elems)[batch_idx];

            __m256 grad_vec = _mm256_set1_ps(grad);
            __m256 max_vec = _mm256_set1_ps(max);

            int base_offset = batch_idx * reduce_size;

            int i = 0;
            for (; i + SIMD_WIDTH - 1 < reduce_size; i += SIMD_WIDTH) {
                __m256 data_vec = _mm256_loadu_ps(((float*)in->data->elems) + base_offset + i);
                __m256 mask = _mm256_cmp_ps(data_vec, max_vec, _CMP_EQ_OQ);
                __m256 grad_contrib = _mm256_and_ps(grad_vec, mask);
                __m256 in_grad = _mm256_loadu_ps(((float*)in->grad->elems) + base_offset + i);
                __m256 new_grad = _mm256_add_ps(in_grad, grad_contrib);
                _mm256_storeu_ps(((float*)in->grad->elems) + base_offset + i, new_grad);
            }

            for (; i < reduce_size; ++i) {
                if (((float*)in->data->elems)[base_offset + i] == max) {
                    ((float*)in->grad->elems)[base_offset + i] += grad;
                }
            }
        }
    }
}

void sum_full_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: sum_full_grad_op: Computing gradient for full sum reduction");

    Tensor* in = prev[0];

    if (!in->requires_grad) {
        return;
    }

    // out is a scalar, so out->grad->elems[0] contains the gradient
    float output_grad = ((float*)out->grad->elems)[0];
    int in_size = numel(in->shape, in->ndim);

    if (is_contiguous(in)) {
        // Fast path: contiguous input tensor
        __m256 grad_vec = _mm256_set1_ps(output_grad);

        int i = 0;
        for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH) {
            __m256 in_grad = _mm256_loadu_ps(((float*)in->grad->elems) + i);
            __m256 new_grad = _mm256_add_ps(in_grad, grad_vec);
            _mm256_storeu_ps(((float*)in->grad->elems) + i, new_grad);
        }

        // Handle remaining elements
        for (; i < in_size; ++i) {
            ((float*)in->grad->elems)[i] += output_grad;
        }
    } else {
        // Slow path: non-contiguous input tensor
        int* in_strides = in->strides;
        int* in_shape = in->shape;
        int in_ndim = in->ndim;

        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx) {
            int temp_idx = linear_idx;
            int in_offset = 0;

            // Convert linear index to multidimensional coordinates and compute offset
            for (int d = in_ndim - 1; d >= 0; --d) {
                int coord = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
                in_offset += coord * in_strides[d];
            }

            ((float*)in->grad->elems)[in_offset] += output_grad;
        }
    }
}

void mean_full_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: mean_full_grad_op: Computing gradient for full "
             "mean reduction");

    Tensor* in = prev[0];

    if (!in->requires_grad) {
        return;
    }

    float output_grad = ((float*)out->grad->elems)[0];
    int in_size = numel(in->shape, in->ndim);

    float scaled_grad = output_grad / in_size;

    if (is_contiguous(in)) {
        __m256 grad_vec = _mm256_set1_ps(scaled_grad);

        int i = 0;
        for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH) {
            __m256 in_grad = _mm256_loadu_ps(((float*)in->grad->elems) + i);
            __m256 new_grad = _mm256_add_ps(in_grad, grad_vec);
            _mm256_storeu_ps(((float*)in->grad->elems) + i, new_grad);
        }

        for (; i < in_size; ++i) {
            ((float*)in->grad->elems)[i] += scaled_grad;
        }
    } else {
        int* in_strides = in->strides;
        int* in_shape = in->shape;
        int in_ndim = in->ndim;

        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx) {
            int temp_idx = linear_idx;
            int in_offset = 0;

            for (int d = in_ndim - 1; d >= 0; --d) {
                int coord = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
                in_offset += coord * in_strides[d];
            }

            ((float*)in->grad->elems)[in_offset] += scaled_grad;
        }
    }
}

void max_full_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    LOG_INFO("GRAD: max_full_grad_op: Computing gradient for full max "
             "reduction");
    Tensor* in = prev[0];

    if (!in->requires_grad) {
        return;
    }

    float output_grad = ((float*)out->grad->elems)[0];
    float max_val = ((float*)out->data->elems)[0];
    int in_size = numel(in->shape, in->ndim);

    if (is_contiguous(in)) {
        __m256 grad_vec = _mm256_set1_ps(output_grad);
        __m256 max_vec = _mm256_set1_ps(max_val);

        int i = 0;
        for (; i + SIMD_WIDTH - 1 < in_size; i += SIMD_WIDTH) {
            __m256 data_vec = _mm256_loadu_ps(((float*)in->data->elems) + i);
            __m256 mask = _mm256_cmp_ps(data_vec, max_vec, _CMP_EQ_OQ);
            __m256 grad_contrib = _mm256_and_ps(grad_vec, mask);
            __m256 in_grad = _mm256_loadu_ps(((float*)in->grad->elems) + i);
            __m256 new_grad = _mm256_add_ps(in_grad, grad_contrib);
            _mm256_storeu_ps(((float*)in->grad->elems) + i, new_grad);
        }

        for (; i < in_size; ++i) {
            if (((float*)in->data->elems)[i] == max_val) {
                ((float*)in->grad->elems)[i] += output_grad;
            }
        }
    } else {
        int* in_strides = in->strides;
        int* in_shape = in->shape;
        int in_ndim = in->ndim;

        for (int linear_idx = 0; linear_idx < in_size; ++linear_idx) {
            int temp_idx = linear_idx;
            int in_offset = 0;

            for (int d = in_ndim - 1; d >= 0; --d) {
                int coord = temp_idx % in_shape[d];
                temp_idx /= in_shape[d];
                in_offset += coord * in_strides[d];
            }

            if (((float*)in->data->elems)[in_offset] == max_val) {
                ((float*)in->grad->elems)[in_offset] += output_grad;
            }
        }
    }
}

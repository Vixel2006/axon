#include "utils.h"
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "logger.h"
#include "ops/reduction_ops.h"

#define SIMD_WIDTH 8

void sum_op(Tensor* a, Tensor* out, int axis, bool keepdim) {
    LOG_INFO("OP: sum_op: Performing sum reduction along axis %d (keepdim=%d)", axis, keepdim);

    if (axis < 0 || axis >= a->ndim) {
        LOG_ERROR("sum_op: Invalid axis %d for tensor with %d dimensions", axis, a->ndim);
        return;
    }

    size_t size = numel(out->shape, out->ndim);
    float* data = malloc(sizeof(float) * size);
    memset(data, 0, size * sizeof(float));

    // Calculate dimensions
    size_t outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= a->shape[i];
    }

    size_t reduction_size = a->shape[axis];

    size_t inner_size = 1;
    for (int i = axis + 1; i < a->ndim; i++) {
        inner_size *= a->shape[i];
    }

    size_t reduction_stride = a->strides[axis];

    // Process each outer x inner slice
    for (size_t outer = 0; outer < outer_size; outer++) {
        for (size_t inner = 0; inner < inner_size; inner++) {

            // Calculate base input offset for this (outer, inner) position
            size_t input_base = 0;

            // Add outer dimension offsets
            size_t temp_outer = outer;
            for (int d = axis - 1; d >= 0; d--) {
                size_t coord = temp_outer % a->shape[d];
                temp_outer /= a->shape[d];
                input_base += coord * a->strides[d];
            }

            // Add inner dimension offsets
            size_t temp_inner = inner;
            for (int d = a->ndim - 1; d > axis; d--) {
                size_t coord = temp_inner % a->shape[d];
                temp_inner /= a->shape[d];
                input_base += coord * a->strides[d];
            }

            // Calculate output offset
            size_t output_offset = 0;
            int out_dim_idx = 0;

            // Map outer coordinates to output
            temp_outer = outer;
            for (int d = 0; d < axis; d++) {
                size_t stride_product = 1;
                for (int k = d + 1; k < axis; k++) {
                    stride_product *= a->shape[k];
                }
                size_t coord = temp_outer / stride_product;
                temp_outer %= stride_product;
                output_offset += coord * out->strides[out_dim_idx++];
            }

            // Handle keepdim case
            if (keepdim) {
                out_dim_idx++; // Skip the reduced dimension (always 0)
            }

            // Map inner coordinates to output
            temp_inner = inner;
            for (int d = axis + 1; d < a->ndim; d++) {
                size_t stride_product = 1;
                for (int k = d + 1; k < a->ndim; k++) {
                    stride_product *= a->shape[k];
                }
                size_t coord = temp_inner / stride_product;
                temp_inner %= stride_product;
                output_offset += coord * out->strides[out_dim_idx++];
            }

            // Perform reduction along the specified axis
            float sum = 0.0f;

            if (reduction_stride == 1 && reduction_size >= SIMD_WIDTH) {
                // SIMD path for contiguous data
                __m256 sum_vec = _mm256_setzero_ps();
                size_t i = 0;

                for (; i + SIMD_WIDTH - 1 < reduction_size; i += SIMD_WIDTH) {
                    __m256 vec_a = _mm256_loadu_ps(a->data->data + input_base + i);
                    sum_vec = _mm256_add_ps(sum_vec, vec_a);
                }

                // Horizontal sum of SIMD vector
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                __m128 lo = _mm256_extractf128_ps(sum_vec, 0);
                __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
                __m128 total = _mm_add_ps(lo, hi);
                sum = _mm_cvtss_f32(total);

                // Handle remaining elements
                for (; i < reduction_size; i++) {
                    sum += a->data->data[input_base + i * reduction_stride];
                }
            } else {
                // Scalar path
                for (size_t i = 0; i < reduction_size; i++) {
                    sum += a->data->data[input_base + i * reduction_stride];
                }
            }

            data[output_offset] = sum;
        }
    }
    from_data(out, data);
    free(data);
}

void mean_op(Tensor* a, Tensor* out, int axis, bool keepdim) {
    LOG_INFO("OP: mean_op: Performing mean reduction along axis %d (keepdim=%d)", axis, keepdim);

    if (axis < 0 || axis >= a->ndim) {
        LOG_ERROR("mean_op: Invalid axis %d for tensor with %d dimensions", axis, a->ndim);
        return;
    }

    // First compute the sum
    sum_op(a, out, axis, keepdim);

    // Then divide by the reduction size
    float reduction_size = (float)a->shape[axis];
    size_t size = numel(out->shape, out->ndim);

    for (size_t i = 0; i < size; i++) {
        out->data->data[i] /= reduction_size;
    }
}

void max_op(Tensor* a, Tensor* out, int axis, bool keepdim) {
    LOG_INFO("OP: max_op: Performing max reduction along axis %d (keepdim=%d)", axis, keepdim);

    if (axis < 0 || axis >= a->ndim) {
        LOG_ERROR("max_op: Invalid axis %d for tensor with %d dimensions", axis, a->ndim);
        return;
    }

    size_t size = numel(out->shape, out->ndim);
    float* data = malloc(sizeof(float) * size);
    for (size_t i = 0; i < size; i++) {
        data[i] = -FLT_MAX;
    }

    // Calculate dimensions
    size_t outer_size = 1;
    for (int i = 0; i < axis; i++) {
        outer_size *= a->shape[i];
    }

    size_t reduction_size = a->shape[axis];

    size_t inner_size = 1;
    for (int i = axis + 1; i < a->ndim; i++) {
        inner_size *= a->shape[i];
    }

    size_t reduction_stride = a->strides[axis];

    // Process each outer x inner slice
    for (size_t outer = 0; outer < outer_size; outer++) {
        for (size_t inner = 0; inner < inner_size; inner++) {

            // Calculate base input offset for this (outer, inner) position
            size_t input_base = 0;

            // Add outer dimension offsets
            size_t temp_outer = outer;
            for (int d = axis - 1; d >= 0; d--) {
                size_t coord = temp_outer % a->shape[d];
                temp_outer /= a->shape[d];
                input_base += coord * a->strides[d];
            }

            // Add inner dimension offsets
            size_t temp_inner = inner;
            for (int d = a->ndim - 1; d > axis; d--) {
                size_t coord = temp_inner % a->shape[d];
                temp_inner /= a->shape[d];
                input_base += coord * a->strides[d];
            }

            // Calculate output offset
            size_t output_offset = 0;
            int out_dim_idx = 0;

            // Map outer coordinates to output
            temp_outer = outer;
            for (int d = axis - 1; d >= 0; d--) {
                size_t coord = temp_outer % a->shape[d];
                temp_outer /= a->shape[d];
                output_offset += coord * out->strides[out_dim_idx++];
            }

            // Handle keepdim case
            if (keepdim) {
                out_dim_idx++; // Skip the reduced dimension (always 0)
            }

            // Map inner coordinates to output
            temp_inner = inner;
            for (int d = axis + 1; d < a->ndim; d++) {
                size_t coord = temp_inner % a->shape[d];
                temp_inner /= a->shape[d];
                output_offset += coord * out->strides[out_dim_idx++];
            }

            // Find maximum along the specified axis
            float max_val = -FLT_MAX;

            if (reduction_stride == 1 && reduction_size >= SIMD_WIDTH) {
                // SIMD path for contiguous data
                __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
                size_t i = 0;

                for (; i + SIMD_WIDTH - 1 < reduction_size; i += SIMD_WIDTH) {
                    __m256 vec_a = _mm256_loadu_ps(a->data->data + input_base + i);
                    max_vec = _mm256_max_ps(max_vec, vec_a);
                }

                // Horizontal max of SIMD vector
                __m128 vlow = _mm256_castps256_ps128(max_vec);
                __m128 vhigh = _mm256_extractf128_ps(max_vec, 1);
                vlow = _mm_max_ps(vlow, vhigh);
                vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1)));
                vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2)));
                max_val = _mm_cvtss_f32(vlow);

                // Handle remaining elements
                for (; i < reduction_size; i++) {
                    max_val = fmaxf(max_val, a->data->data[input_base + i * reduction_stride]);
                }
            } else {
                // Scalar path
                for (size_t i = 0; i < reduction_size; i++) {
                    max_val = fmaxf(max_val, a->data->data[input_base + i * reduction_stride]);
                }
            }

            data[output_offset] = max_val;
        }
    }
    from_data(out, data);
    free(data);
}

void sum_full_op(Tensor* a, Tensor* out) {
    LOG_INFO("OP: sum_full_op: Performing full sum reduction");

    size_t total_elements = numel(a->shape, a->ndim);

    bool is_a_contiguous = is_contiguous(a);

    float* total_sum = malloc(sizeof(float));
    total_sum[0] = 0.0f;

    if (is_a_contiguous) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + (SIMD_WIDTH - 1) < total_elements; i += SIMD_WIDTH) {
            __m256 vec_a = _mm256_loadu_ps(a->data->data + i);
            sum_vec = _mm256_add_ps(sum_vec, vec_a);
        }

        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 lo_half = _mm256_extractf128_ps(sum_vec, 0);
        __m128 hi_half = _mm256_extractf128_ps(sum_vec, 1);
        __m128 total_sum_m128 = _mm_add_ps(lo_half, hi_half);
        total_sum[0] = _mm_cvtss_f32(total_sum_m128);

        for (; i < total_elements; ++i) {
            total_sum[0] += a->data->data[i];
        }
    } else {
        int* coords = malloc(a->ndim * sizeof(int));
        if (!coords) {
            LOG_ERROR("sum_full_op: Failed to allocate memory for coords.");
            free(total_sum);
            return;
        }

        memset(coords, 0, a->ndim * sizeof(int));

        for (size_t elem = 0; elem < total_elements; ++elem) {
            size_t flat_idx = 0;
            for (int d = 0; d < a->ndim; ++d) {
                flat_idx += (size_t)coords[d] * a->strides[d];
            }

            total_sum[0] += a->data->data[flat_idx];

            int carry = 1;
            for (int d = a->ndim - 1; d >= 0 && carry; --d) {
                coords[d] += carry;
                if (coords[d] < a->shape[d]) {
                    carry = 0;
                } else {
                    coords[d] = 0;
                    carry = 1;
                }
            }
        }

        free(coords);
    }

    from_data(out, total_sum);
    free(total_sum);
}

void mean_full_op(Tensor* a, Tensor* out) {
    LOG_INFO("OP: mean_full_op: Performing full mean reduction");

    sum_full_op(a, out);

    size_t total_elements = numel(a->shape, a->ndim);

    if (total_elements > 0) {
        out->data->data[0] /= total_elements;
    } else {
        if (out) {
            out->data->data[0] = 0.0f;
        }
    }
}

void max_full_op(Tensor* a, Tensor* out) {
    LOG_INFO("OP: max_full_op: Performing full max reduction");

    size_t total_elements = numel(a->shape, a->ndim);
    float* data = malloc(sizeof(float) * total_elements);

    if (total_elements == 0) {
        data[0] = -FLT_MAX;
        out->requires_grad = a->requires_grad;
        return;
    }

    bool is_a_contiguous = is_contiguous(a);

    float max_val = -FLT_MAX;

    if (is_a_contiguous) {
        __m256 max_vec = _mm256_set1_ps(-FLT_MAX);
        size_t i = 0;

        for (; i + (SIMD_WIDTH - 1) < total_elements; i += SIMD_WIDTH) {
            __m256 vec_a = _mm256_loadu_ps(a->data->data + i);
            max_vec = _mm256_max_ps(max_vec, vec_a);
        }

        __m128 vlow = _mm256_castps256_ps128(max_vec);
        __m128 vhigh = _mm256_extractf128_ps(max_vec, 1);
        vlow = _mm_max_ps(vlow, vhigh);
        vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1)));
        vlow = _mm_max_ps(vlow, _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 0, 3, 2)));
        max_val = _mm_cvtss_f32(vlow);

        for (; i < total_elements; ++i) {
            max_val = fmaxf(max_val, a->data->data[i]);
        }
    } else {
        int* coords = malloc(a->ndim * sizeof(int));
        if (!coords) {
            LOG_ERROR("max_full_op: Failed to allocate memory for coords.");
            free(data);
            return;
        }

        memset(coords, 0, a->ndim * sizeof(int));
        bool first = true;

        for (size_t elem = 0; elem < total_elements; ++elem) {
            size_t flat_idx = 0;
            for (int d = 0; d < a->ndim; ++d) {
                flat_idx += (size_t)coords[d] * a->strides[d];
            }

            if (first) {
                max_val = a->data->data[flat_idx];
                first = false;
            } else {
                max_val = fmaxf(max_val, a->data->data[flat_idx]);
            }

            int carry = 1;
            for (int d = a->ndim - 1; d >= 0 && carry; --d) {
                coords[d] += carry;
                if (coords[d] < a->shape[d]) {
                    carry = 0;
                } else {
                    coords[d] = 0;
                    carry = 1;
                }
            }
        }

        free(coords);
    }

    data[0] = max_val;

    from_data(out, data);
    free(data);
}

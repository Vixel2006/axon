#include "autograd/autograd_movement.h"

#define SIMD_WIDTH 8

void concat_grad_op(Tensor* out, Tensor** prev, int n_prev, void* extras) {
    int offset = 0;

    LOG_INFO("Starting concat_grad_op: out.numel=%d, n_prev=%d", numel(out->shape, out->ndim), n_prev);

    for (int tensor_idx = 0; tensor_idx < n_prev; ++tensor_idx) {
        int size = numel(prev[tensor_idx]->shape, prev[tensor_idx]->ndim);

        LOG_INFO("Processing tensor %d: size=%d, ndim=%d, requires_grad=%d, offset=%d", tensor_idx, size, prev[tensor_idx]->ndim, prev[tensor_idx]->requires_grad, offset);

        if (prev[tensor_idx]->requires_grad) {
            if (!is_contiguous(prev[tensor_idx])) {
                LOG_WARN("Tensor %d is not contiguous, using strided accumulation", tensor_idx);

                for (int linear = 0; linear < size; ++linear) {
                    int idx = linear;
                    int in_offset = 0;

                    for (int d = prev[tensor_idx]->ndim - 1; d >= 0; --d) {
                        int coord = idx % prev[tensor_idx]->shape[d];
                        idx /= prev[tensor_idx]->shape[d];
                        in_offset += coord * prev[tensor_idx]->strides[d];
                    }

                    prev[tensor_idx]->grad->data->data[in_offset] += out->grad->data->data[offset + linear] * prev[tensor_idx]->data->data[in_offset];

                    if (linear < 5) {
                        LOG_INFO("    [non-contig] linear=%d -> in_offset=%d, grad=%f", linear, in_offset, prev[tensor_idx]->grad->data->data[in_offset]);
                    }
                }
            } else {
                LOG_INFO("Tensor %d is contiguous, using SIMD if possible", tensor_idx);

                int i = 0;
                for (; i + SIMD_WIDTH - 1 < size; i += SIMD_WIDTH) {
                    __m256 din = _mm256_loadu_ps(prev[tensor_idx]->grad->data->data + i);
                    __m256 dout = _mm256_loadu_ps(out->grad->data->data + i + offset);
                    __m256 dd = _mm256_add_ps(din, dout);

                    _mm256_store_ps(prev[tensor_idx]->grad->data->data + i, dd);
                }

                for (; i < size; ++i) {
                    prev[tensor_idx]->grad->data->data[i] += out->grad->data->data[i + offset];

                    if (i < 5) { // show first few elems
                        LOG_INFO("    [contig] i=%d, grad=%f", i, prev[tensor_idx]->grad->data->data[i]);
                    }
                }
            }

            offset += size;
            LOG_INFO("Finished tensor %d, new offset=%d", tensor_idx, offset);
        }
    }

    LOG_INFO("Finished concat_grad_op");
}


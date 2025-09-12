#include "utils.h"
#include <stdlib.h>
#include <string.h> // For memcpy

#include "logger.h"
#include "ops/ops.h"
#include "ops/ops_utils.h"

void view_op(Tensor* in, Tensor* out, int* shape, int ndim) {
    LOG_INFO("OP: view_op: Creating view from Tensor %p (ndim=%d)", (void*)in, ndim);
    LOG_INFO("OP: view_op: Input shape: ");
    print_shape(in->shape, in->ndim);

    if (ndim > 0) {
        out->strides = malloc(ndim * sizeof(int));
        if (!out->strides) {
            LOG_ERROR("view_op: Failed to allocate memory for strides array.");
            free(out->shape);
            out->shape = NULL;
            return;
        }
        out->strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            out->strides[i] = out->strides[i + 1] * out->shape[i + 1];
        }
    } else {
        out->strides = NULL;
    }

    tensor_init_view(out, in);

    LOG_INFO("OP: view_op: Output shape: ");
    print_shape(out->shape, out->ndim);
}

void unsqueeze_op(Tensor* in, Tensor* out, int dim) {
    LOG_INFO("OP: unsqueeze_op: Unsqueezing Tensor %p at dimension %d", (void*)in, dim);
    LOG_INFO("OP: unsqueeze_op: Input shape: ");
    print_shape(in->shape, in->ndim);

    if (dim < 0 || dim > in->ndim) {
        LOG_ERROR("unsqueeze_op: Invalid dimension %d for unsqueeze operation "
                  "(ndim=%d).",
                  dim, in->ndim);
        return;
    }

    out->strides = malloc(out->ndim * sizeof(int));
    if (!out->strides) {
        LOG_ERROR("unsqueeze_op: Failed to allocate memory for strides array.");
        free(out->shape);
        return;
    }

    // Compute strides based on input tensor's strides
    for (int i = 0; i < out->ndim; ++i) {
        if (i < dim)
            out->strides[i] = in->strides[i];
        else if (i == dim)
            out->strides[i] = (i < out->ndim - 1) ? out->strides[i + 1] * out->shape[i + 1] : 1;
        else
            out->strides[i] = in->strides[i - 1];
    }

    // Share data and gradients
    tensor_init_view(out, in);

    LOG_INFO("OP: unsqueeze_op: Output shape: ");
    print_shape(out->shape, out->ndim);
}

void squeeze_op(Tensor* in, Tensor* out, int dim) {
    LOG_INFO("OP: squeeze_op: Squeezing Tensor %p at dimension %d", (void*)in, dim);
    LOG_INFO("OP: squeeze_op: Input shape: ");
    print_shape(in->shape, in->ndim);

    if (dim < 0 || dim >= in->ndim || in->shape[dim] != 1) {
        LOG_ERROR("squeeze_op: Invalid dimension %d for squeeze operation "
                  "(ndim=%d, shape[%d]=%d). Dimension must be 1.",
                  dim, in->ndim, dim, in->shape[dim]);
        return;
    }

    out->strides = malloc(out->ndim * sizeof(int));
    if (!out->shape || !out->strides) {
        LOG_ERROR("squeeze_op: Failed to allocate memory for shape or strides "
                  "array.");
        free(out->shape);
        free(out->strides);
        return;
    }

    for (int i = 0; i < out->ndim; ++i) {
        out->strides[i] = (i < dim) ? in->strides[i] : in->strides[i + 1];
    }

    // Share data and gradients
    tensor_init_view(out, in);

    LOG_INFO("OP: squeeze_op: Output shape: ");
    print_shape(out->shape, out->ndim);
}

void transpose_op(Tensor* in, Tensor* out, int N, int M) {
    if (!in || !out) {
        LOG_ERROR("transpose_op: Input or output tensor is NULL.");
        return;
    }

    LOG_INFO("OP: transpose_op: Transposing Tensor %p (dims %d, %d)", (void*)in, N, M);
    LOG_INFO("OP: transpose_op: Input shape: ");
    print_shape(in->shape, in->ndim);

    if (N < 0 || N >= in->ndim || M < 0 || M >= in->ndim || N == M) {
        LOG_ERROR("transpose_op: Invalid dimensions N=%d or M=%d for transpose operation "
                  "(ndim=%d). N and M must be within bounds and different.",
                  N, M, in->ndim);
        return;
    }

    out->strides = malloc(out->ndim * sizeof(int));

    if (!out->shape || !out->strides) {
        LOG_ERROR("transpose_op: Failed to allocate memory for shape or strides "
                  "array.");
        free(out->shape);
        free(out->strides);
        out->shape = NULL;
        out->strides = NULL;
        out->ndim = 0;
        return;
    }

    for (int i = 0; i < out->ndim; ++i) {
        if (i == N) {
            out->strides[i] = in->strides[M];
        } else if (i == M) {
            out->strides[i] = in->strides[N];
        } else {
            out->strides[i] = in->strides[i];
        }
    }

    // Share data and gradients
    tensor_init_view(out, in);

    LOG_INFO("OP: transpose_op: Output shape: ");
    print_shape(out->shape, out->ndim);
}

void expand_op(Tensor* in, Tensor* out, const int* shape) {
    LOG_INFO("OP: expand_op: Expanding Tensor %p", (void*)in);
    LOG_INFO("OP: expand_op: Input shape: ");
    print_shape(in->shape, in->ndim);

    out->strides = malloc(out->ndim * sizeof(int));
    if (!out->shape || !out->strides) {
        LOG_ERROR("expand_op: Failed to allocate memory for shape or strides array.");
        free(out->shape);
        free(out->strides);
        return;
    }

    for (int i = 0; i < in->ndim; ++i) {
        if (in->shape[i] != 1 && in->shape[i] != shape[i]) {
            LOG_ERROR("expand_op: Cannot expand dimension %d from %d to %d. "
                      "Dimension must be 1 or match target size.",
                      i, in->shape[i], shape[i]);
            free(out->shape);
            free(out->strides);
            return;
        }
        out->strides[i] = (in->shape[i] == 1) ? 0 : in->strides[i];
    }

    // Share data and gradients
    tensor_init_view(out, in);

    LOG_INFO("OP: expand_op: Output shape: ");
    print_shape(out->shape, out->ndim);
}

void broadcast_op(Tensor* in, Tensor* out, int ndim, const int* shape) {
    LOG_INFO("OP: broadcast_op: Broadcasting Tensor %p to ndim=%d", (void*)in, ndim);
    LOG_INFO("OP: broadcast_op: Input shape: ");
    print_shape(in->shape, in->ndim);

    // Error checking for null tensors
    if (!in || !out || !shape) {
        LOG_ERROR("broadcast_op ERROR: Input tensor, output tensor, or shape "
                  "array is NULL! in=%p, out=%p, shape=%p",
                  (void*)in, (void*)out, (void*)shape);
        return;
    }

    if (!out->shape || !out->strides) {
        LOG_ERROR("broadcast_op: Failed to allocate memory for shape or strides "
                  "array.");
        free(out->shape);
        free(out->strides);
        return;
    }

    int in_dim = in->ndim - 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (in_dim >= 0) {
            if (in->shape[in_dim] == shape[i]) {
                out->strides[i] = in->strides[in_dim];
            } else if (in->shape[in_dim] == 1) {
                out->strides[i] = 0;
            } else {
                LOG_ERROR("broadcast_op: Cannot broadcast dimension %d from %d to "
                          "%d. Dimension must be 1 or match target size.",
                          in_dim, in->shape[in_dim], shape[i]);
                free(out->shape);
                free(out->strides);
                return;
            }
            in_dim--;
        } else {
            out->strides[i] = 0;
        }
    }

    // Share data and gradients
    tensor_init_view(out, in);

    LOG_INFO("OP: broadcast_op: Output shape: ");
    print_shape(out->shape, out->ndim);
}

void concat_op(Tensor** in, Tensor* out, int num_tensors, int axis) {
    LOG_INFO("OP: concat_op: Concatenating %d tensors along axis %d", num_tensors, axis);

    if (!out || !out->data->elems) {
        LOG_ERROR("concat_op: Output tensor or its data is NULL.");
        return;
    }

    // Get the total number of elements in the output tensor
    int total_out_elements = numel(out->shape, out->ndim);

    // Iterate through each element of the output tensor
    int* out_coords = (int*)malloc(out->ndim * sizeof(int));
    if (!out_coords) {
        LOG_ERROR("concat_op: Failed to allocate memory for out_coords.");
        return;
    }

    // Iterate through each element of the output tensor
    for (int k = 0; k < total_out_elements; ++k) {
        // Convert linear index k to multi-dimensional index (out_coords)
        int temp_k = k;
        for (int d = out->ndim - 1; d >= 0; --d) {
            out_coords[d] = temp_k / out->strides[d];
            temp_k %= out->strides[d];
        }

        // Determine which input tensor this element belongs to
        int current_input_tensor_idx = 0;
        int current_axis_val_in_out = out_coords[axis]; // Value of the coordinate along the concatenation
                                                        // axis in the output tensor
        int current_offset_in_axis = 0;                 // Offset within the concatenation axis for the current input tensor

        for (int t_idx = 0; t_idx < num_tensors; ++t_idx) {
            Tensor* t = in[t_idx];
            if (current_axis_val_in_out < current_offset_in_axis + t->shape[axis]) {
                current_input_tensor_idx = t_idx;
                break;
            }
            current_offset_in_axis += t->shape[axis];
        }

        Tensor* src_tensor = in[current_input_tensor_idx];

        // Calculate the corresponding index in the source tensor
        int* src_coords = (int*)malloc(src_tensor->ndim * sizeof(int));
        if (!src_coords) {
            LOG_ERROR("concat_op: Failed to allocate memory for src_coords.");
            free(out_coords);
            return;
        }

        for (int d = 0; d < out->ndim; ++d) {
            if (d == axis) {
                src_coords[d] = out_coords[d] - current_offset_in_axis;
            } else {
                src_coords[d] = out_coords[d];
            }
        }

        // Convert multi-dimensional src_coords to linear index in src_tensor
        int src_linear_idx = 0;
        for (int d = 0; d < src_tensor->ndim; ++d) {
            src_linear_idx += src_coords[d] * src_tensor->strides[d];
        }

        // Copy the element
        ((float*)out->data->elems)[k] = ((float*)src_tensor->data->elems)[src_linear_idx];

        free(src_coords);
    }
    free(out_coords);
    LOG_INFO("OP: concat_op: Concatenation complete.");
}

void stack_op(Tensor** in, Tensor* out, int num_tensors, int axis) {
}

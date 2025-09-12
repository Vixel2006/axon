#include "ops/ops_utils.h"
#include "logger.h"
#include "utils.h"

int tensor_alloc_shape(int ndim, const int* shape, int** out_shape) {
    LOG_INFO("DEBUG: tensor_alloc_shape: Allocating shape array for ndim=%d", ndim);
    if (!*out_shape) {
        LOG_ERROR("tensor_alloc_shape: out_shape is NULL.");
        return -1;
    }
    memcpy(*out_shape, shape, ndim * sizeof(int));
    LOG_INFO("DEBUG: tensor_alloc_shape: Successfully allocated shape array at %p", (void*)*out_shape);
    return 0;
}

int tensor_alloc_strides(int ndim, const int* strides, int** out_strides) {
    LOG_INFO("DEBUG: tensor_alloc_strides: Allocating strides array for ndim=%d", ndim);
    if (!*out_strides) {
        LOG_ERROR("tensor_alloc_strides: out_strides is NULL.");
        return -1;
    }
    memcpy(*out_strides, strides, ndim * sizeof(int));
    LOG_INFO("DEBUG: tensor_alloc_strides: Successfully allocated strides array at %p", (void*)*out_strides);
    return 0;
}

int tensor_copy_layout(Tensor* in, Tensor* out, const int* shape) {
    LOG_INFO("DEBUG: tensor_copy_layout: Copying layout from Tensor %p to %p", (void*)in, (void*)out);
    out->ndim = in->ndim;
    if (tensor_alloc_shape(out->ndim, shape, &out->shape) != 0) {
        LOG_ERROR("tensor_copy_layout: Failed to allocate shape "
                  "for out tensor");
        return -1;
    }
    out->strides = NULL;
    LOG_INFO("DEBUG: tensor_copy_layout: Successfully copied layout");
    return 0;
}

void reference_shared_ptr(shared_ptr* out, shared_ptr in) {
    LOG_INFO("DEBUG: reference_shared_ptr: Referencing SharedPtr from %p to %p", (void*)in.elems, (void*)out);
    if (in.elems == NULL) {
        *out = (shared_ptr){0};
        LOG_INFO("DEBUG: reference_shared_ptr: Input SharedPtr is NULL, setting output to NULL");
        return;
    }

    *out = in;
    (*out).ref_counter++;
    LOG_INFO("DEBUG: reference_shared_ptr: Incremented ref_counter for SharedPtr %p to %d", (void*)(*out).elems, (*out).ref_counter);
}

void tensor_init_view(Tensor* out, Tensor* in) {
    LOG_INFO("DEBUG: tensor_init_view: Initializing view from Tensor %p to %p", (void*)in, (void*)out);
    // Input validation
    if (!out || !in) {
        LOG_ERROR("tensor_init_view: Invalid input (out or in is NULL).");
        return;
    }

    reference_shared_ptr(out->data, *in->data);
    reference_shared_ptr(out->grad, *in->grad);

    out->requires_grad = in->requires_grad;
    LOG_INFO("DEBUG: tensor_init_view: Successfully initialized view");
}

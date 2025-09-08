#include "ops/ops_utils.h"
#include "utils.h"

int tensor_alloc_shape(int ndim, const int *shape, int **out_shape) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] tensor_alloc_shape: Allocating shape array for ndim=%d\n",
      ndim);
  if (!*out_shape) {
    DEBUG_PRINT("[IDRAK_DEBUG] tensor_alloc_shape: out_shape is NULL\n");
    return -1;
  }
  memcpy(*out_shape, shape, ndim * sizeof(int));
  DEBUG_PRINT("[IDRAK_DEBUG] tensor_alloc_shape: Successfully allocated shape "
              "array at %p\n",
              (void *)*out_shape);
  return 0;
}

int tensor_alloc_strides(int ndim, const int *strides, int **out_strides) {
  DEBUG_PRINT("[IDRAK_DEBUG] tensor_alloc_strides: Allocating strides array "
              "for ndim=%d\n",
              ndim);
  if (!*out_strides) {
    DEBUG_PRINT("[IDRAK_DEBUG] tensor_alloc_strides: out_strides is NULL\n");
    return -1;
  }
  memcpy(*out_strides, strides, ndim * sizeof(int));
  DEBUG_PRINT("[IDRAK_DEBUG] tensor_alloc_strides: Successfully allocated "
              "strides array at %p\n",
              (void *)*out_strides);
  return 0;
}

int tensor_copy_layout(Tensor *in, Tensor *out, const int *shape) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] tensor_copy_layout: Copying layout from Tensor %p to %p\n",
      (void *)in, (void *)out);
  out->ndim = in->ndim;
  if (tensor_alloc_shape(out->ndim, shape, &out->shape) != 0) {
    DEBUG_PRINT("[IDRAK_DEBUG] tensor_copy_layout: Failed to allocate shape "
                "for out tensor\n");
    return -1;
  }
  out->strides = NULL;
  DEBUG_PRINT("[IDRAK_DEBUG] tensor_copy_layout: Successfully copied layout\n");
  return 0;
}

void reference_shared_ptr(SharedPtr **out, SharedPtr *in) {
  DEBUG_PRINT("[IDRAK_DEBUG] reference_shared_ptr: Referencing SharedPtr from "
              "%p to %p\n",
              (void *)in, (void *)*out);
  if (!in) {
    *out = NULL;
    DEBUG_PRINT("[IDRAK_DEBUG] reference_shared_ptr: Input SharedPtr is NULL, "
                "setting output "
                "to NULL\n");
    return;
  }

  *out = in;
  in->ref_counter++;
  DEBUG_PRINT("[IDRAK_DEBUG] reference_shared_ptr: Incremented ref_counter for "
              "SharedPtr %p to %d\n",
              (void *)in, in->ref_counter);
}

void tensor_init_view(Tensor *out, Tensor *in) {
  DEBUG_PRINT("[IDRAK_DEBUG] tensor_init_view: Initializing view from Tensor "
              "%p to %p\n",
              (void *)in, (void *)out);
  // Input validation
  if (!out || !in) {
    DEBUG_PRINT(
        "[IDRAK_DEBUG] tensor_init_view: Invalid input (out or in is NULL)\n");
    return;
  }

  reference_shared_ptr(&out->data, in->data);
  reference_shared_ptr(&out->grad, in->grad);

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
  DEBUG_PRINT(
      "[IDRAK_DEBUG] tensor_init_view: Successfully initialized view\n");
}

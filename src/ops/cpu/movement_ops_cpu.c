#include <stdio.h>
#include <stdlib.h>

#include "ops.h"

/**
 * @brief Reshapes a tensor view without copying data.
 *
 * Creates a new view `out` of tensor `in` with the given shape.
 * The underlying `data` and `grad` buffers are shared between
 * `in` and `out`. Only metadata (shape, strides) is updated.
 *
 * @param in     Input tensor.
 * @param out    Output tensor (allocated by caller).
 * @param shape  New shape array.
 * @param ndim   Number of dimensions in the new shape.
 *
 * @effects Allocates and sets `out->shape` and `out->strides`.
 * @effects Shares `in->data` and `in->grad` with `out`.
 * @note No copy is performed. Caller must ensure the new shape is valid.
 */
void view_op(Tensor *in, Tensor *out, int *shape, int ndim) {
  out->owns_data = false;
  out->ndim = ndim;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    return;
  }

  for (int i = 0; i < out->ndim; ++i) {
    out->shape[i] = shape[i];
  }

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides) {
    free(out->shape);  // Fixed: Free allocated shape before returning
    out->shape = NULL;
    return;
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

/**
 * @brief Adds a dimension of size 1 at the specified position.
 *
 * Creates a view of `in` with an extra dimension of size 1
 * inserted at position `dim`.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller).
 * @param dim  Dimension index where the new axis is inserted.
 *
 * @effects Allocates and sets `out->shape` and `out->strides`.
 * @effects Shares `in->data` and `in->grad` with `out`.
 * @note No data copy, only metadata change.
 */
void unsqueeze_op(Tensor *in, Tensor *out, int dim) {
  // Validate dimension index
  if (dim < 0 || dim > in->ndim) {
    return;
  }

  out->owns_data = false;
  out->ndim = in->ndim + 1;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    return;
  }

  for (int i = 0; i < out->ndim; ++i) {
    if (i < dim) {
      out->shape[i] = in->shape[i];
    } else if (i == dim) {
      out->shape[i] = 1;
    } else {
      out->shape[i] = in->shape[i - 1];
    }
  }

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides) {
    free(out->shape);  // Fixed: Free allocated shape before calling free_tensor
    out->shape = NULL;
    free_tensor(out);
    return;
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

/**
 * @brief Removes a dimension of size 1 at the specified position.
 *
 * Creates a view of `in` with the dimension at index `dim` removed.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller).
 * @param dim  Dimension index to remove (must be of size 1).
 *
 * @effects Allocates and sets `out->shape` and `out->strides`.
 * @effects Shares `in->data` and `in->grad` with `out`.
 * @note Caller must ensure the removed dimension is actually size 1.
 */
void squeeze_op(Tensor *in, Tensor *out, int dim) {
  // Validate dimension index and size
  if (dim < 0 || dim >= in->ndim || in->shape[dim] != 1) {
    return;
  }

  out->owns_data = false;
  out->ndim = in->ndim - 1;

  // Handle edge case where we're removing the last dimension
  if (out->ndim == 0) {
    out->shape = NULL;
    out->strides = NULL;
  } else {
    out->shape = malloc(out->ndim * sizeof(int));
    if (!out->shape) {
      return;
    }

    for (int i = 0; i < out->ndim; ++i) {
      if (i < dim) {
        out->shape[i] = in->shape[i];
      } else {
        out->shape[i] = in->shape[i + 1];
      }
    }

    out->strides = compute_strides(out->shape, out->ndim);
    if (!out->strides) {
      free(out->shape);  // Fixed: Free allocated shape before calling
                         // free_tensor
      out->shape = NULL;
      free_tensor(out);
      return;
    }
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

/**
 * @brief Transposes two dimensions of a tensor.
 *
 * Creates a view of `in` with dimensions `N` and `M` swapped.
 *
 * @param in   Input tensor.
 * @param out  Output tensor (allocated by caller).
 * @param N    First dimension index.
 * @param M    Second dimension index.
 *
 * @effects Allocates and sets `out->shape` and `out->strides`.
 * @effects Shares `in->data` and `in->grad` with `out`.
 * @note Only metadata is updated, no copy is performed.
 */
void transpose_op(Tensor *in, Tensor *out, int N, int M) {
  // Validate dimension indices
  if (N < 0 || N >= in->ndim || M < 0 || M >= in->ndim) {
    return;
  }

  out->owns_data = false;
  out->ndim = in->ndim;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    return;
  }

  // Copy shape with dimensions swapped
  for (int i = 0; i < out->ndim; ++i) {
    if (i == N) {
      out->shape[i] = in->shape[M];
    } else if (i == M) {
      out->shape[i] = in->shape[N];
    } else {
      out->shape[i] = in->shape[i];
    }
  }

  // For transpose, we need to manually compute strides to reflect the swap
  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->strides) {
    free(out->shape);
    out->shape = NULL;
    return;
  }

  // Copy strides with dimensions swapped
  for (int i = 0; i < out->ndim; ++i) {
    if (i == N) {
      out->strides[i] = in->strides[M];
    } else if (i == M) {
      out->strides[i] = in->strides[N];
    } else {
      out->strides[i] = in->strides[i];
    }
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

/**
 * @brief Expands a tensor to a larger shape using broadcasting rules.
 *
 * Creates a view of `in` with expanded dimensions according to `shape`.
 * Dimensions of size 1 can be broadcast to larger sizes, other dimensions
 * must match.
 *
 * @param in     Input tensor.
 * @param out    Output tensor (allocated by caller).
 * @param shape  Target shape array (same ndim as `in`).
 *
 * @effects Allocates and sets `out->shape` and `out->strides`.
 * @effects Shares `in->data` and `in->grad` with `out`.
 * @note Expansion uses stride=0 for broadcasted dims.
 */
void expand_op(Tensor *in, Tensor *out, const int *shape) {
  out->owns_data = false;
  out->ndim = in->ndim;

  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    return;
  }

  // Copy the target shape
  for (int i = 0; i < out->ndim; ++i) {
    out->shape[i] = shape[i];
  }

  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->strides) {
    free(out->shape);
    out->shape = NULL;
    return;
  }

  // Set strides: 0 for broadcasted dimensions, original stride otherwise
  for (int i = 0; i < in->ndim; ++i) {
    // Validate that non-unit dimensions match
    if (in->shape[i] != 1 && in->shape[i] != shape[i]) {
      free(out->shape);
      free(out->strides);
      out->shape = NULL;
      out->strides = NULL;
      return;
    }
    out->strides[i] = (in->shape[i] == 1) ? 0 : in->strides[i];
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

/**
 * @brief Broadcasts a tensor to a new shape with potentially more dimensions.
 *
 * Creates a view of `in` broadcast to the target shape. The input tensor
 * is broadcast according to NumPy broadcasting rules.
 *
 * @param in     Input tensor.
 * @param out    Output tensor (allocated by caller).
 * @param ndim   Number of dimensions in target shape.
 * @param shape  Target shape array.
 *
 * @effects Allocates and sets `out->shape` and `out->strides`.
 * @effects Shares `in->data` and `in->grad` with `out`.
 */
void broadcast_op(Tensor *in, Tensor *out, int ndim, const int *shape) {
  out->owns_data = false;
  out->ndim = ndim;

  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    return;  // Fixed: Don't call free_tensor here as it expects a valid tensor
  }

  for (int i = 0; i < out->ndim; ++i) {
    out->shape[i] = shape[i];
  }

  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->strides) {
    free(out->shape);
    out->shape = NULL;
    return;
  }

  // Implement broadcasting logic
  int in_dim = in->ndim - 1;
  for (int i = out->ndim - 1; i >= 0; --i) {
    if (in_dim >= 0) {
      if (in->shape[in_dim] == shape[i]) {
        // Dimension sizes match, use original stride
        out->strides[i] = in->strides[in_dim];
      } else if (in->shape[in_dim] == 1) {
        // Input dimension is 1, broadcast with stride 0
        out->strides[i] = 0;
      } else {
        // Invalid broadcast: dimensions don't match and input isn't 1
        free(out->shape);
        free(out->strides);
        out->shape = NULL;
        out->strides = NULL;
        return;
      }
      in_dim--;
    } else {
      // New dimension added on the left, stride is 0
      out->strides[i] = 0;
    }
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

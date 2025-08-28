#include <stdio.h>

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
    free_tensor(out);
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
    free_tensor(out);
    out->shape = NULL;
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
  out->owns_data = false;
  out->ndim = in->ndim - 1;
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
    free_tensor(out);
    out->shape = NULL;
    return;
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
  out->owns_data = false;
  out->ndim = in->ndim;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    return;
  }

  for (int i = 0; i < out->ndim; ++i) {
    if (i == N) {
      out->shape[i] = in->shape[M];
    } else if (i == M) {
      out->shape[i] = in->shape[N];
    } else {
      out->shape[i] = in->shape[i];
    }
  }

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides) {
    free_tensor(out);
    out->shape = NULL;
    return;
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
  out->shape = in->shape;

  out->strides = malloc(out->ndim * sizeof(int));

  for (int i = 0; i < in->ndim; ++i) {
    out->strides[i] = in->strides[i] == shape[i] ? in->strides[i] : 0;
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

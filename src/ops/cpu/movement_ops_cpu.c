#include <stdio.h>
#include <stdlib.h>

#include "ops/ops.h"
#include "ops/ops_utils.h"

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
  if (tensor_copy_layout(in, out, shape) != 0) return;
  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides) {
    free(out->shape);
    out->shape = NULL;
    return;
  }
  tensor_init_view(out, in);
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
  if (dim < 0 || dim > in->ndim) return;

  out->ndim = in->ndim + 1;
  int *new_shape = malloc(out->ndim * sizeof(int));
  if (!new_shape) return;

  for (int i = 0; i < out->ndim; ++i) {
    if (i < dim)
      new_shape[i] = in->shape[i];
    else if (i == dim)
      new_shape[i] = 1;
    else
      new_shape[i] = in->shape[i - 1];
  }

  if (tensor_alloc_shape(out->ndim, new_shape, &out->shape) != 0) {
    free(new_shape);
    return;
  }
  free(new_shape);

  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides) {
    free(out->shape);
    return;
  }

  tensor_init_view(out, in);
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
  if (dim < 0 || dim >= in->ndim || in->shape[dim] != 1) return;

  out->ndim = in->ndim - 1;
  if (out->ndim == 0) {
    out->shape = NULL;
    out->strides = NULL;
  } else {
    int *new_shape = malloc(out->ndim * sizeof(int));
    if (!new_shape) return;
    for (int i = 0; i < out->ndim; ++i) {
      new_shape[i] = (i < dim) ? in->shape[i] : in->shape[i + 1];
    }
    if (tensor_alloc_shape(out->ndim, new_shape, &out->shape) != 0) {
      free(new_shape);
      return;
    }
    free(new_shape);

    out->strides = compute_strides(out->shape, out->ndim);
    if (!out->strides) {
      free(out->shape);
      return;
    }
  }

  tensor_init_view(out, in);
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
  if (N < 0 || N >= in->ndim || M < 0 || M >= in->ndim) return;

  out->ndim = in->ndim;
  int *new_shape = malloc(out->ndim * sizeof(int));
  int *new_strides = malloc(out->ndim * sizeof(int));
  if (!new_shape || !new_strides) {
    free(new_shape);
    free(new_strides);
    return;
  }

  for (int i = 0; i < out->ndim; ++i) {
    if (i == N) {
      new_shape[i] = in->shape[M];
      new_strides[i] = in->strides[M];
    } else if (i == M) {
      new_shape[i] = in->shape[N];
      new_strides[i] = in->strides[N];
    } else {
      new_shape[i] = in->shape[i];
      new_strides[i] = in->strides[i];
    }
  }

  out->shape = new_shape;
  out->strides = new_strides;
  tensor_init_view(out, in);
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
  out->ndim = in->ndim;
  if (tensor_alloc_shape(out->ndim, shape, &out->shape) != 0) return;

  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->strides) {
    free(out->shape);
    return;
  }

  for (int i = 0; i < in->ndim; ++i) {
    if (in->shape[i] != 1 && in->shape[i] != shape[i]) {
      free(out->shape);
      free(out->strides);
      return;
    }
    out->strides[i] = (in->shape[i] == 1) ? 0 : in->strides[i];
  }
  tensor_init_view(out, in);
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
  out->ndim = ndim;
  if (tensor_alloc_shape(ndim, shape, &out->shape) != 0) return;
  out->strides = malloc(ndim * sizeof(int));
  if (!out->strides) {
    free(out->shape);
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
        free(out->shape);
        free(out->strides);
        return;
      }
      in_dim--;
    } else {
      out->strides[i] = 0;
    }
  }
  tensor_init_view(out, in);
}

#include <stdio.h>
#include <stdlib.h>

#include "ops/ops.h"
#include "ops/ops_utils.h"

void view_op(Tensor *in, Tensor *out, int *shape, int ndim) {
  if (tensor_copy_layout(in, out, shape) != 0)
    return;
  out->strides = compute_strides(out->shape, out->ndim);
  if (!out->strides) {
    free(out->shape);
    out->shape = NULL;
    return;
  }
  tensor_init_view(out, in);
}

void unsqueeze_op(Tensor *in, Tensor *out, int dim) {
  if (dim < 0 || dim > in->ndim)
    return;

  out->ndim = in->ndim + 1;
  int *new_shape = malloc(out->ndim * sizeof(int));
  if (!new_shape)
    return;

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

void squeeze_op(Tensor *in, Tensor *out, int dim) {
  if (dim < 0 || dim >= in->ndim || in->shape[dim] != 1)
    return;

  out->ndim = in->ndim - 1;
  if (out->ndim == 0) {
    out->shape = NULL;
    out->strides = NULL;
  } else {
    int *new_shape = malloc(out->ndim * sizeof(int));
    if (!new_shape)
      return;
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

void transpose_op(Tensor *in, Tensor *out, int N, int M) {
  if (N < 0 || N >= in->ndim || M < 0 || M >= in->ndim)
    return;

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

void expand_op(Tensor *in, Tensor *out, const int *shape) {
  out->ndim = in->ndim;
  if (tensor_alloc_shape(out->ndim, shape, &out->shape) != 0)
    return;

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

void broadcast_op(Tensor *in, Tensor *out, int ndim, const int *shape) {
  out->ndim = ndim;
  if (tensor_alloc_shape(ndim, shape, &out->shape) != 0)
    return;
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

void concat_op(Tensor **in, Tensor *out, int num_tensors, int axis) {
  int ndim = in[0]->ndim;

  out->ndim = ndim;

  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    free_tensor(out);
    return;
  }

  for (int dim = 0; dim < out->ndim; ++dim) {
    if (dim == axis) {
      out->shape[dim] = 0;
      for (int i = 0; i < num_tensors; ++i) {
        out->shape[dim] += in[i]->shape[dim];
      }
    } else {
      out->shape[dim] = in[0]->shape[dim];
    }
  }

  out->strides = compute_strides(out->shape, out->ndim);
  int total_size = numel(out->shape, out->ndim);

  out->data = malloc(total_size * sizeof(float));
  out->grad = malloc(total_size * sizeof(float));

  if (!out->data || !out->grad) {
    free_tensor(out);
    return;
  }

  out->requires_grad = false;

  int axis_offset = 0;

  for (int tensor_idx = 0; tensor_idx < num_tensors; ++tensor_idx) {
    out->requires_grad = out->requires_grad || in[tensor_idx]->requires_grad;

    int tensor_size = numel(in[tensor_idx]->shape, in[tensor_idx]->ndim);

    for (int i = 0; i < tensor_size; ++i) {
      int *in_coords = malloc(in[tensor_idx]->ndim * sizeof(int));
      int temp_i = i;
      for (int d = in[tensor_idx]->ndim - 1; d >= 0; --d) {
        in_coords[d] = temp_i % in[tensor_idx]->shape[d];
        temp_i /= in[tensor_idx]->shape[d];
      }

      int *out_coords = malloc(out->ndim * sizeof(int));
      for (int d = 0; d < out->ndim; ++d) {
        if (d == axis) {
          out_coords[d] = in_coords[d] + axis_offset;
        } else {
          out_coords[d] = in_coords[d];
        }
      }

      int out_idx = 0;
      for (int d = 0; d < out->ndim; ++d) {
        out_idx += out_coords[d] * out->strides[d];
      }

      out->data[out_idx] = in[tensor_idx]->data[i];
      if (in[tensor_idx]->grad) {
        out->grad[out_idx] = in[tensor_idx]->grad[i];
      } else {
        out->grad[out_idx] = 0.0f;
      }

      free(in_coords);
      free(out_coords);
    }
    axis_offset += in[tensor_idx]->shape[axis];
  }

  out->owns_data = true;
}

void stack_op(Tensor **in, Tensor *out, int num_tensors, int axis) {
  out->ndim = in[0]->ndim + 1;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape) {
    free_tensor(out);
    return;
  }

  // Calculate output shape
  for (int i = 0; i < out->ndim; ++i) {
    if (i < axis) {
      out->shape[i] = in[0]->shape[i];
    } else if (i == axis) {
      out->shape[i] = num_tensors;
    } else {
      out->shape[i] = in[0]->shape[i - 1];
    }
  }

  out->strides = compute_strides(out->shape, out->ndim);
  int total_size = numel(out->shape, out->ndim);

  out->data = malloc(total_size * sizeof(float));
  out->grad = malloc(total_size * sizeof(float));
  if (!out->data || !out->grad) {
    free_tensor(out);
    return;
  }

  out->requires_grad = false;

  for (int tensor_idx = 0; tensor_idx < num_tensors; ++tensor_idx) {
    out->requires_grad = out->requires_grad || in[tensor_idx]->requires_grad;

    // Calculate total number of elements in input tensor
    int input_numel = numel(in[tensor_idx]->shape, in[tensor_idx]->ndim);

    // Iterate through all possible multi-dimensional indices
    int *coords = calloc(in[tensor_idx]->ndim, sizeof(int));

    for (int linear_idx = 0; linear_idx < input_numel; ++linear_idx) {
      // Calculate input index using strides
      int input_idx = 0;
      for (int d = 0; d < in[tensor_idx]->ndim; ++d) {
        input_idx += coords[d] * in[tensor_idx]->strides[d];
      }

      // Create output coordinates by inserting the stack dimension
      int *out_coords = malloc(out->ndim * sizeof(int));
      for (int d = 0; d < out->ndim; ++d) {
        if (d < axis) {
          out_coords[d] = coords[d];
        } else if (d == axis) {
          out_coords[d] = tensor_idx;
        } else {
          out_coords[d] = coords[d - 1];
        }
      }

      // Calculate output index using strides
      int output_idx = 0;
      for (int d = 0; d < out->ndim; ++d) {
        output_idx += out_coords[d] * out->strides[d];
      }

      // Copy data and gradients
      out->data[output_idx] = in[tensor_idx]->data[input_idx];
      if (in[tensor_idx]->grad) {
        out->grad[output_idx] = in[tensor_idx]->grad[input_idx];
      } else {
        out->grad[output_idx] = 0.0f;
      }

      free(out_coords);

      // Increment coordinates (like an odometer)
      int carry = 1;
      for (int d = in[tensor_idx]->ndim - 1; d >= 0 && carry; --d) {
        coords[d] += carry;
        if (coords[d] >= in[tensor_idx]->shape[d]) {
          coords[d] = 0;
          carry = 1;
        } else {
          carry = 0;
        }
      }
    }

    free(coords);
  }

  out->owns_data = true;
}

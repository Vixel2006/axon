#include "autograd/autograd.h"
#include "utils.h"
#include <stdlib.h>

typedef struct {
  int axis;
} stackExtras;

typedef struct {
  int axis;
} concatExtras;

void stack_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  IDRAK_DEBUG("GRAD ",
      "stack_grad_op: Computing gradient for stack operation\n");
  IDRAK_DEBUG("GRAD ", "stack_grad_op: Output gradient (out) shape: ");
  print_shape(out->shape, out->ndim);
  for (int idx = 0; idx < n_prev; ++idx) {
    IDRAK_DEBUG("GRAD ", "stack_grad_op: Input tensor %d (prev[%d]) shape: ", idx, idx);
    print_shape(prev[idx]->shape, prev[idx]->ndim);
  }


  stackExtras *stack_extras = (stackExtras *)extras;
  int axis = stack_extras->axis;

  for (int idx = 0; idx < n_prev; ++idx) {
    Tensor *input_tensor = prev[idx];
    int input_tensor_numel = numel(input_tensor->shape, input_tensor->ndim);

    for (int j = 0; j < input_tensor_numel; ++j) {
      // Convert linear index j to multi-dimensional coordinates for
      // input_tensor
      int *input_coords = malloc(input_tensor->ndim * sizeof(int));
      int temp_j = j;
      for (int d = input_tensor->ndim - 1; d >= 0; --d) {
        input_coords[d] = temp_j % input_tensor->shape[d];
        temp_j /= input_tensor->shape[d];
      }

      // Convert input_coords to multi-dimensional coordinates for out_tensor
      // by inserting the stack dimension
      int *out_coords = malloc(out->ndim * sizeof(int));
      for (int d = 0; d < out->ndim; ++d) {
        if (d < axis) {
          out_coords[d] = input_coords[d];
        } else if (d == axis) {
          out_coords[d] =
              idx; // The index of the current input tensor in the stack
        } else {
          out_coords[d] = input_coords[d - 1];
        }
      }

      // Calculate linear index in out_tensor
      int out_linear_idx = 0;
      for (int d = 0; d < out->ndim; ++d) {
        out_linear_idx += out_coords[d] * out->strides[d];
      }

      // Accumulate gradient
      input_tensor->grad->ptr[j] += out->grad->ptr[out_linear_idx];

      free(input_coords);
      free(out_coords);
    }
  }
}

void concat_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  IDRAK_DEBUG("GRAD ", "concat_grad_op: Computing gradient for "
              "concatenate operation\n");
  IDRAK_DEBUG("GRAD ", "concat_grad_op: Output gradient (out) shape: ");
  print_shape(out->shape, out->ndim);
  for (int i = 0; i < n_prev; ++i) {
    IDRAK_DEBUG("GRAD ", "concat_grad_op: Input tensor %d (prev[%d]) shape: ", i, i);
    print_shape(prev[i]->shape, prev[i]->ndim);
  }

  concatExtras *concat_extras = (concatExtras *)extras;
  int axis = concat_extras->axis;

  int current_offset = 0; // Offset in the output gradient tensor

  for (int i = 0; i < n_prev; ++i) {
    Tensor *input_tensor = prev[i];

    // Calculate the size of the current input tensor's slice along the concat
    // axis
    int slice_size_along_axis = input_tensor->shape[axis];

    int input_tensor_numel = numel(input_tensor->shape, input_tensor->ndim);

    for (int j = 0; j < input_tensor_numel; ++j) {
      // Convert linear index j to multi-dimensional coordinates for
      // input_tensor
      int *input_coords = malloc(input_tensor->ndim * sizeof(int));
      int temp_j = j;
      for (int d = input_tensor->ndim - 1; d >= 0; --d) {
        input_coords[d] = temp_j % input_tensor->shape[d];
        temp_j /= input_tensor->shape[d];
      }

      // Convert input_coords to multi-dimensional coordinates for out_tensor
      int *out_coords = malloc(out->ndim * sizeof(int));
      for (int d = 0; d < out->ndim; ++d) {
        if (d == axis) {
          out_coords[d] = input_coords[d] + current_offset;
        } else {
          out_coords[d] = input_coords[d];
        }
      }

      // Calculate linear index in out_tensor
      int out_linear_idx = 0;
      for (int d = 0; d < out->ndim; ++d) {
        out_linear_idx += out_coords[d] * out->strides[d];
      }

      // Accumulate gradient
      input_tensor->grad->ptr[j] += out->grad->ptr[out_linear_idx];

      free(input_coords);
      free(out_coords);
    }

    current_offset += slice_size_along_axis;
  }
}
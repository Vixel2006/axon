#include "utils.h"
#include <stdlib.h>

#include "ops/ops.h"
#include "ops/ops_utils.h"

static void cleanup_tensor_for_view(Tensor *out) {
  DEBUG_PRINT("[IDRAK_DEBUG] cleanup_tensor_for_view: Cleaning up tensor %p "
              "for view operation\n",
              (void *)out);

  if (out->data) {
    if (out->data->ref_counter <= 1) {
      free(out->data->ptr);
      free(out->data);
    } else {
      out->data->ref_counter--;
    }
  }

  if (out->grad) {
    if (out->grad->ref_counter <= 1) {
      free(out->grad->ptr);
      free(out->grad);
    } else {
      out->grad->ref_counter--;
    }
  }

  if (out->strides) {
    free(out->strides);
  }

  if (out->shape) {
    free(out->shape);
  }
}

void view_op(Tensor *in, Tensor *out, int *shape, int ndim) {
  DEBUG_PRINT("[IDRAK_DEBUG] view_op: Creating view from Tensor %p (ndim=%d)\n",
              (void *)in, ndim);

  cleanup_tensor_for_view(out);

  out->ndim = ndim;
  out->shape = malloc(ndim * sizeof(int));
  if (!out->shape)
    return;

  for (int i = 0; i < ndim; ++i) {
    out->shape[i] = shape[i];
  }

  if (ndim > 0) {
    out->strides = malloc(ndim * sizeof(int));
    if (!out->strides) {
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

  if (in->data) {
    out->data = in->data;
    in->data->ref_counter++;
  } else {
    out->data = NULL;
  }

  if (in->grad) {
    out->grad = in->grad;
    in->grad->ref_counter++;
  } else {
    out->grad = NULL;
  }

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void unsqueeze_op(Tensor *in, Tensor *out, int dim) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] unsqueeze_op: Unsqueezing Tensor %p at dimension %d\n",
      (void *)in, dim);

  if (dim < 0 || dim > in->ndim)
    return;

  cleanup_tensor_for_view(out);

  out->ndim = in->ndim + 1;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape)
    return;

  for (int i = 0; i < out->ndim; ++i) {
    if (i < dim)
      out->shape[i] = in->shape[i];
    else if (i == dim)
      out->shape[i] = 1;
    else
      out->shape[i] = in->shape[i - 1];
  }

  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->strides) {
    free(out->shape);
    return;
  }

  // Compute strides based on input tensor's strides
  for (int i = 0; i < out->ndim; ++i) {
    if (i < dim)
      out->strides[i] = in->strides[i];
    else if (i == dim)
      out->strides[i] =
          (i < out->ndim - 1) ? out->strides[i + 1] * out->shape[i + 1] : 1;
    else
      out->strides[i] = in->strides[i - 1];
  }

  // Share data and gradients
  if (in->data) {
    out->data = in->data;
    in->data->ref_counter++;
  }
  if (in->grad) {
    out->grad = in->grad;
    in->grad->ref_counter++;
  }

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void squeeze_op(Tensor *in, Tensor *out, int dim) {
  DEBUG_PRINT("[IDRAK_DEBUG] squeeze_op: Squeezing Tensor %p at dimension %d\n",
              (void *)in, dim);

  if (dim < 0 || dim >= in->ndim || in->shape[dim] != 1)
    return;

  cleanup_tensor_for_view(out);

  out->ndim = in->ndim - 1;
  if (out->ndim == 0) {
    out->shape = NULL;
    out->strides = NULL;
  } else {
    out->shape = malloc(out->ndim * sizeof(int));
    out->strides = malloc(out->ndim * sizeof(int));
    if (!out->shape || !out->strides) {
      free(out->shape);
      free(out->strides);
      return;
    }

    for (int i = 0; i < out->ndim; ++i) {
      out->shape[i] = (i < dim) ? in->shape[i] : in->shape[i + 1];
      out->strides[i] = (i < dim) ? in->strides[i] : in->strides[i + 1];
    }
  }

  // Share data and gradients
  if (in->data) {
    out->data = in->data;
    in->data->ref_counter++;
  }
  if (in->grad) {
    out->grad = in->grad;
    in->grad->ref_counter++;
  }

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void transpose_op(Tensor *in, Tensor *out, int N, int M) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] transpose_op: Transposing Tensor %p (dims %d, %d)\n",
      (void *)in, N, M);

  if (N < 0 || N >= in->ndim || M < 0 || M >= in->ndim)
    return;

  cleanup_tensor_for_view(out);

  out->ndim = in->ndim;
  out->shape = malloc(out->ndim * sizeof(int));
  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->shape || !out->strides) {
    free(out->shape);
    free(out->strides);
    return;
  }

  for (int i = 0; i < out->ndim; ++i) {
    if (i == N) {
      out->shape[i] = in->shape[M];
      out->strides[i] = in->strides[M];
    } else if (i == M) {
      out->shape[i] = in->shape[N];
      out->strides[i] = in->strides[N];
    } else {
      out->shape[i] = in->shape[i];
      out->strides[i] = in->strides[i];
    }
  }

  // Share data and gradients
  if (in->data) {
    out->data = in->data;
    in->data->ref_counter++;
  }
  if (in->grad) {
    out->grad = in->grad;
    in->grad->ref_counter++;
  }

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void expand_op(Tensor *in, Tensor *out, const int *shape) {
  DEBUG_PRINT("[IDRAK_DEBUG] expand_op: Expanding Tensor %p\n", (void *)in);

  cleanup_tensor_for_view(out);

  out->ndim = in->ndim;
  out->shape = malloc(out->ndim * sizeof(int));
  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->shape || !out->strides) {
    free(out->shape);
    free(out->strides);
    return;
  }

  for (int i = 0; i < in->ndim; ++i) {
    if (in->shape[i] != 1 && in->shape[i] != shape[i]) {
      free(out->shape);
      free(out->strides);
      return;
    }
    out->shape[i] = shape[i];
    out->strides[i] = (in->shape[i] == 1) ? 0 : in->strides[i];
  }

  // Share data and gradients
  if (in->data) {
    out->data = in->data;
    in->data->ref_counter++;
  }
  if (in->grad) {
    out->grad = in->grad;
    in->grad->ref_counter++;
  }

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void broadcast_op(Tensor *in, Tensor *out, int ndim, const int *shape) {
  DEBUG_PRINT("[IDRAK_DEBUG] broadcast_op: Broadcasting Tensor %p to ndim=%d\n",
              (void *)in, ndim);

  cleanup_tensor_for_view(out);

  out->ndim = ndim;
  out->shape = malloc(ndim * sizeof(int));
  out->strides = malloc(ndim * sizeof(int));
  if (!out->shape || !out->strides) {
    free(out->shape);
    free(out->strides);
    return;
  }

  int in_dim = in->ndim - 1;
  for (int i = ndim - 1; i >= 0; --i) {
    out->shape[i] = shape[i];
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

  // Share data and gradients
  if (in->data) {
    out->data = in->data;
    in->data->ref_counter++;
  }
  if (in->grad) {
    out->grad = in->grad;
    in->grad->ref_counter++;
  }

  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

// Zero-copy concat: Creates a view that references multiple tensors
// Note: This requires a special SharedPtr that can handle multiple data sources
void concat_op(Tensor **in, Tensor *out, int num_tensors, int axis) {
  DEBUG_PRINT(
      "[IDRAK_DEBUG] concat_op: Concatenating %d tensors along axis %d\n",
      num_tensors, axis);

  cleanup_tensor_for_view(out);

  int ndim = in[0]->ndim;
  out->ndim = ndim;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape)
    return;

  // Calculate output shape
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
  if (!out->strides) {
    free(out->shape);
    return;
  }

  // For zero-copy concat, we need a special data structure
  // This is a simplified version - in practice, you'd need a more complex
  // SharedPtr that can handle multiple source tensors with offset mapping

  // For now, we'll create a virtual view that shares the first tensor's data
  // and use custom stride calculations for access
  if (in[0]->data) {
    out->data = in[0]->data;
    in[0]->data->ref_counter++;

    // Store references to other tensors (this would need custom implementation)
    // out->concat_sources = malloc(num_tensors * sizeof(Tensor*));
    // for (int i = 0; i < num_tensors; ++i) {
    //     out->concat_sources[i] = in[i];
    // }
  }

  if (in[0]->grad) {
    out->grad = in[0]->grad;
    in[0]->grad->ref_counter++;
  }

  out->requires_grad = false;
  for (int i = 0; i < num_tensors; ++i) {
    out->requires_grad = out->requires_grad || in[i]->requires_grad;
  }
  out->grad_fn = NULL;
}

// Zero-copy stack: Creates a view with adjusted strides
void stack_op(Tensor **in, Tensor *out, int num_tensors, int axis) {
  DEBUG_PRINT("[IDRAK_DEBUG] stack_op: Stacking %d tensors along axis %d\n",
              num_tensors, axis);

  cleanup_tensor_for_view(out);

  out->ndim = in[0]->ndim + 1;
  out->shape = malloc(out->ndim * sizeof(int));
  if (!out->shape)
    return;

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

  out->strides = malloc(out->ndim * sizeof(int));
  if (!out->strides) {
    free(out->shape);
    return;
  }

  // Compute strides for the stacked view
  // This creates a virtual view where each tensor appears at regular intervals
  int tensor_size = numel(in[0]->shape, in[0]->ndim);

  for (int i = 0; i < out->ndim; ++i) {
    if (i < axis) {
      out->strides[i] = in[0]->strides[i] * num_tensors;
    } else if (i == axis) {
      out->strides[i] = tensor_size;
    } else {
      out->strides[i] = in[0]->strides[i - 1] * num_tensors;
    }
  }

  // Share data from first tensor (in practice, you'd need custom handling)
  if (in[0]->data) {
    out->data = in[0]->data;
    in[0]->data->ref_counter++;
  }

  if (in[0]->grad) {
    out->grad = in[0]->grad;
    in[0]->grad->ref_counter++;
  }

  out->requires_grad = false;
  for (int i = 0; i < num_tensors; ++i) {
    out->requires_grad = out->requires_grad || in[i]->requires_grad;
  }
  out->grad_fn = NULL;
}

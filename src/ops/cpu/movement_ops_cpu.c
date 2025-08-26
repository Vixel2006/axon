#include "ops.h"
#include <stdio.h>

void view_op(Tensor *in, Tensor *out, int *shape, int ndim) {
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
    free(out->shape);
    out->shape = NULL;
    return;
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void unsqueeze_op(Tensor *in, Tensor *out, int dim) {
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
    free(out->shape);
    out->shape = NULL;
    return;
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void squeeze_op(Tensor *in, Tensor *out, int dim) {
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
    free(out->shape);
    out->shape = NULL;
    return;
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void transpose_op(Tensor *in, Tensor *out, int N, int M) {
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
    free(out->shape);
    out->shape = NULL;
    return;
  }

  out->data = in->data;
  out->grad = in->grad;
  out->requires_grad = in->requires_grad;
  out->grad_fn = NULL;
}

void expand_op(Tensor *in, Tensor *out, const int *shape) {
  out->ndim = in->ndim;
  out->shape = malloc(out->ndim * sizeof(int));

  if (!out->shape) {
    free_tensor(out);
    return;
  }

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

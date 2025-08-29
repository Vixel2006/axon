#include <immintrin.h>

#include "autograd.h"

#define SIMD_WIDTH 8

void sum_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {
  // TODO: We need to define a broadcast function in c.
}

void mean_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {}

void max_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {}
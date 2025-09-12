#include "autograd/autograd.h"
#include "utils.h"
#include <stdlib.h>
#include "logger.h"

typedef struct {
  int axis;
} stackExtras;

typedef struct {
  int axis;
} concatExtras;

void stack_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {}

void concat_grad_op(Tensor *out, Tensor **prev, int n_prev, void *extras) {}
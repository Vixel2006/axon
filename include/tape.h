#ifndef NAWAH_TAPE_H
#define NAWAH_TAPE_H

#include "tensor.h"

typedef struct {
  Tensor *prev;
  void (*backward_fn)(Tensor *out, Tensor *prev, int n_prev);
} Tape;

#endif

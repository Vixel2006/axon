#ifndef NAWAH_NODE_H
#define NAWAH_NODE_H

#include "tensor.h"

typedef struct {
  Tensor *out;
  Tensor **prev;

  int n_prev;
  void *extras;
  void (*backward_fn)(Tensor *out, Tensor **prev, int n_prev, void *extras);
} Node;

Node malloc_node(Tensor *out, Tensor **prev, int n_prev, void *extras,
                 void *backward_fn);

void free_node(Node *p);

#endif

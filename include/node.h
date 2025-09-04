#ifndef IDRAK_NODE_H
#define IDRAK_NODE_H

#include "tensor.h"

typedef struct {
  Tensor *out;
  Tensor **prev;

  int n_prev;
  void *extras;
  void *forward_fn;
  void *backward_fn;
} Node;

Node *malloc_node(Tensor *out, Tensor **prev, int n_prev, void *extras,
                  void *forward_fn, void *backward_fn);

void free_node(Node *p);

#endif

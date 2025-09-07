#include "node.h"

#include <stdlib.h>

Node *malloc_node(Tensor *out, Tensor **prev, int n_prev, void *extras,
                  void *forward_fn, void *backward_fn) {
  Node *node = malloc(sizeof(Node));
  if (!node) {
    return NULL;
  }

  node->n_prev = n_prev;
  node->out = out;
  node->prev = prev;
  node->extras = extras;
  node->forward_fn = forward_fn;
  node->backward_fn = backward_fn;

  return node;
}

void free_node(Node **n) {
  if (n && *n) {
    if ((*n)->extras)
      free((*n)->extras);
    free(*n);
    *n = NULL;
  }
}

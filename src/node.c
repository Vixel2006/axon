#include "node.h"

Node malloc_node(Tensor *out, Tensor **prev, int n_prev, void *extras,
                 void *backward_fn) {
  Node *node = malloc(sizeof(Node));

  node->n_prev = n_prev;
  node->out = out;
  node->prev = prev;
  node->extras = extras;
}

void free_node(Node *n) {
  if (n) {
    if (n->prev) {
      free(n->prev);
      n->prev = NULL;
    }

    if (n->extras) {
      free(n->extras);
      n->extras = NULL;
    }
    free(n);
  }
}

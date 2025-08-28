#include "node.h"

#include <stdlib.h>

/**
 * @brief Initialize a node for the lazy execution plan.
 *
 * @param out         The output tensor from the operation.
 * @param prev        An array of tensors that are inputs to the operation.
 * @param n_prev      The number of input tensors.
 * @param extras      Extra data passed to the forward and backward functions.
 * @param forward_fn  Pointer to the forward function for the operation.
 * @param backward_fn Pointer to the backward function for the operation.
 *
 * @return Pointer to the newly allocated node with the specified shape.
 */
Node *malloc_node(Tensor *out, Tensor **prev, int n_prev, void *extras,
                  void *forward_fn, void *backward_fn) {
  Node *node = malloc(sizeof(Node));
  if (!node) {
    // Handle allocation error
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

/**
 * @brief Free a node and its associated data.
 *
 * @param n  Pointer to the node to free.
 *
 * @note This function releases the memory occupied by the node.
 */
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

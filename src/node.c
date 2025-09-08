#include "node.h"

#include "utils.h"
#include <stdlib.h>

Node *malloc_node(Tensor *out, Tensor **prev, int n_prev, void *extras,
                  void *forward_fn, void *backward_fn) {
  DEBUG_PRINT("[IDRAK_DEBUG] malloc_node: Allocating Node (n_prev=%d, out=%p)\n", n_prev,
              (void *)out);
  Node *node = malloc(sizeof(Node));
  if (!node) {
    DEBUG_PRINT("[IDRAK_DEBUG] malloc_node: Failed to allocate Node\n");
    return NULL;
  }

  node->n_prev = n_prev;
  node->out = out;
  node->prev = prev;
  node->extras = extras;
  node->forward_fn = forward_fn;
  node->backward_fn = backward_fn;

  DEBUG_PRINT("[IDRAK_DEBUG] malloc_node: Successfully allocated Node at %p\n", (void *)node);
  return node;
}

void free_node(Node **n) {
  if (n && *n) {
    DEBUG_PRINT("[IDRAK_DEBUG] free_node: Freeing Node at %p\n", (void *)*n);
    if ((*n)->extras) {
      DEBUG_PRINT("[IDRAK_DEBUG] free_node: Freeing extras for Node at %p\n", (void *)*n);
      free((*n)->extras);
    }
    free(*n);
    *n = NULL;
    DEBUG_PRINT("[IDRAK_DEBUG] free_node: Node at %p successfully freed\n", (void *)*n);
  }
}

#include "node.h"

#include "utils.h"
#include <stdlib.h>

#include "node.h"

#include "utils.h"
#include <stdlib.h>

Node *malloc_node(Tensor *out, Tensor **prev, int n_prev, void *extras,
                  void *forward_fn, void *backward_fn) {
  IDRAK_DEBUG("ALLOC", "malloc_node: Allocating Node (n_prev=%d, out=%p)\n",
              n_prev, (void *)out);
  Node *node = malloc(sizeof(Node));
  if (!node) {
    IDRAK_DEBUG("ALLOC", "malloc_node: Failed to allocate Node\n");
    return NULL;
  }

  node->n_prev = n_prev;
  node->out = out;
  node->prev = prev;
  node->extras = extras;
  node->forward_fn = forward_fn;
  node->backward_fn = backward_fn;

  if (node->out && node->out->data) {
    node->out->data->ref_counter++;
    IDRAK_DEBUG("ALLOC",
                "malloc_node: Incremented ref_counter for out->data to %d\n",
                node->out->data->ref_counter);
  }
  if (node->out && node->out->grad) {
    node->out->grad->ref_counter++;
    IDRAK_DEBUG("ALLOC",
                "malloc_node: Incremented ref_counter for out->grad to %d\n",
                node->out->grad->ref_counter);
  }
  for (int i = 0; i < n_prev; ++i) {
    if (node->prev[i] && node->prev[i]->data) {
      node->prev[i]->data->ref_counter++;
      IDRAK_DEBUG(
          "ALLOC",
          "malloc_node: Incremented ref_counter for prev[%d]->data to %d\n", i,
          node->prev[i]->data->ref_counter);
    }
    if (node->prev[i] && node->prev[i]->grad) {
      node->prev[i]->grad->ref_counter++;
      IDRAK_DEBUG(
          "ALLOC",
          "malloc_node: Incremented ref_counter for prev[%d]->grad to %d\n", i,
          node->prev[i]->grad->ref_counter);
    }
  }

  IDRAK_DEBUG("ALLOC", "malloc_node: Successfully allocated Node at %p\n",
              (void *)node);
  return node;
}

void free_node(Node **n) {
  if (n && *n) {
    IDRAK_DEBUG("FREE ", "free_node: Freeing Node at %p\n", (void *)*n);

    if ((*n)->out && (*n)->out->data) {
      (*n)->out->data->ref_counter--;
      IDRAK_DEBUG("FREE ",
                  "free_node: Decremented ref_counter for out->data to %d\n",
                  (*n)->out->data->ref_counter);
    }
    if ((*n)->out && (*n)->out->grad) {
      (*n)->out->grad->ref_counter--;
      IDRAK_DEBUG("FREE ",
                  "free_node: Decremented ref_counter for out->grad to %d\n",
                  (*n)->out->grad->ref_counter);
    }
    for (int i = 0; i < (*n)->n_prev; ++i) {
      if ((*n)->prev[i] && (*n)->prev[i]->data) {
        (*n)->prev[i]->data->ref_counter--;
        IDRAK_DEBUG(
            "FREE ",
            "free_node: Decremented ref_counter for prev[%d]->data to %d\n", i,
            (*n)->prev[i]->data->ref_counter);
      }
      if ((*n)->prev[i] && (*n)->prev[i]->grad) {
        (*n)->prev[i]->grad->ref_counter--;
        IDRAK_DEBUG(
            "FREE ",
            "free_node: Decremented ref_counter for prev[%d]->grad to %d\n", i,
            (*n)->prev[i]->grad->ref_counter);
      }
    }

    if ((*n)->extras) {
      IDRAK_DEBUG("FREE ", "free_node: Freeing extras for Node at %p\n",
                  (void *)*n);
      free((*n)->extras);
    }
    free(*n);
    *n = NULL;
    IDRAK_DEBUG("FREE ", "free_node: Node at %p successfully freed\n",
                (void *)*n);
  }
}

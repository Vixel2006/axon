#include "node.h"

#include "logger.h"
#include "utils.h"
#include <stdlib.h>

#include "utils.h"
#include <stdlib.h>

Node* nmalloc(Tensor* out, Tensor** prev, int n_prev, void* extras, void* forward_fn, void* backward_fn) {
    LOG_INFO("ALLOC: Allocating Node (n_prev=%d, out=%p)", n_prev, (void*)out);
    Node* node = malloc(sizeof(Node));
    if (!node) {
        LOG_ERROR("ALLOC: Failed to allocate Node");
        return NULL;
    }

    node->n_prev = n_prev;
    node->out = out;
    node->prev = prev;
    node->extras = extras;
    node->forward_fn = forward_fn;
    node->backward_fn = backward_fn;

    if (node->out) {
        node->out->data->ref_counter++;
        LOG_INFO("ALLOC: Incremented ref_counter for out->data to %d", node->out->data->ref_counter);
    }
    if (node->out) {
        node->out->grad->ref_counter++;
        LOG_INFO("ALLOC: Incremented ref_counter for out->grad to %d", node->out->grad->ref_counter);
    }
    for (int i = 0; i < n_prev; ++i) {
        if (node->prev[i]) {
            node->prev[i]->data->ref_counter++;
            LOG_INFO("ALLOC: Incremented ref_counter for prev[%d]->data to %d", i, node->prev[i]->data->ref_counter);
        }
        if (node->prev[i]) {
            node->prev[i]->grad->ref_counter++;
            LOG_INFO("ALLOC: Incremented ref_counter for prev[%d]->grad to %d", i, node->prev[i]->grad->ref_counter);
        }
    }

    LOG_INFO("ALLOC: Successfully allocated Node at %p", (void*)node);
    return node;
}

void nfree(Node* n) {
    if (n) {
        LOG_INFO("FREE: Freeing Node at %p", *n);

        if (n->out) {
            n->out->data->ref_counter--;
            n->out->grad->ref_counter--;
            tfree(n->out);
        }

        for (int i = 0; i < n->n_prev; ++i) {
            if (n->prev[i]) {
                n->prev[i]->data->ref_counter--;
                n->prev[i]->grad->ref_counter--;
                tfree(*n->prev);
            }
        }

        if (n->extras) {
            LOG_INFO("FREE: Freeing extras for Node at %p", *n);
            free(n->extras);
        }

        free(n);
        LOG_INFO("FREE: Node at %p successfully freed", *n);
    }
}

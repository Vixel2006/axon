#ifndef IDRAK_NODE_H
#define IDRAK_NODE_H

#include "tensor.h"

typedef struct {
    Tensor* out;
    Tensor** prev;

    int n_prev;
    void* extras;
    void* forward_fn;
    void* backward_fn;
} Node;

Node* nmalloc(Tensor* out, Tensor** prev, int n_prev, void* extras, void* forward_fn, void* backward_fn);

void nfree(Node* p);

#endif

#ifndef IDRAK_UTILS_H
#define IDRAK_UTILS_H

#include "tensor.h"
#include <stdio.h>

// Define DEBUG to enable debug prints
#define DEBUG 1

#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...) \
    do { \
        fprintf(stderr, fmt, ##__VA_ARGS__); \
    } while (0)
#else
#define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif


  int get_num_batches(const int *shape, int ndim);
int get_flat_index(const Tensor *t, const int *indices);

#endif

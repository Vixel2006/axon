#ifndef IDRAK_UTILS_H
#define IDRAK_UTILS_H

#include "tensor.h"
#include <stdio.h>

extern int _idrak_debug_enabled;

#define DEBUG_PRINT(fmt, ...) \
    do { \
        if (_idrak_debug_enabled) { \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

int get_num_batches(const int *shape, int ndim);
int get_flat_index(const Tensor *t, const int *indices);
void idrak_set_debug_mode(int enable);

#endif


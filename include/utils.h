#ifndef UTILS_H
#define UTILS_H

#include "tensor.h"
#include <stdio.h>

// ANSI color codes
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

// Debugging
extern int _idrak_debug_enabled;
void idrak_set_debug_mode(int enable);

// Macro for debug printing
#define DEBUG_PRINT(color, fmt, ...) \
    if (_idrak_debug_enabled) { \
        fprintf(stderr, color "[DEBUG] %s:%d:%s(): " fmt ANSI_COLOR_RESET, \
                __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    }

// Define IDRAK_DEBUG, IDRAK_ERROR, and IDRAK_WARNING
#define IDRAK_DEBUG(tag, fmt, ...) DEBUG_PRINT(ANSI_COLOR_CYAN, "[%s] " fmt, tag, ##__VA_ARGS__)
#define IDRAK_ERROR(fmt, ...) DEBUG_PRINT(ANSI_COLOR_RED, "[ERROR] " fmt, ##__VA_ARGS__)
#define IDRAK_WARNING(fmt, ...) DEBUG_PRINT(ANSI_COLOR_YELLOW, "[WARNING] " fmt, ##__VA_ARGS__)


int get_num_batches(const int *shape, int ndim);
int get_flat_index(const Tensor *t, const int *indices);
void print_shape(const int *shape, int ndim);

#endif // UTILS_H



#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H

#include <stdbool.h>
#include <stdlib.h>

#define SAFE_FREE(ptr_addr, free_func)                                                             \
    do                                                                                             \
    {                                                                                              \
        if (*(ptr_addr))                                                                           \
        {                                                                                          \
            free_func(*(ptr_addr));                                                                \
            *(ptr_addr) = NULL;                                                                    \
        }                                                                                          \
    } while (0)

typedef enum
{
    CPU,
    CUDA
} Device;

typedef struct
{
    float* data;
    int size;
    int counter;
} Storage;

Storage* smalloc(float* data, int size);
void sfree(Storage* s);

typedef struct Tensor
{
    Storage* data;
    Storage* grad;
    int* shape;
    int* strides;
    Device device;
    int ndim;
    bool requires_grad;
} Tensor;

Tensor* tmalloc(int* shape, int ndim, Device device, bool requires_grad);
void tfree(Tensor* t);

void gmalloc(Tensor* t, float init);
void gfree(Tensor* t);

int numel(const int* shape, int ndim);
int* compute_strides(const int* shape, int ndim);
bool is_contiguous(Tensor* t);
bool shapes_equal(const int* shape1, int ndim1, const int* shape2, int ndim2);
#endif

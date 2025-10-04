#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H

#include <stdbool.h>
#include <stdlib.h>

#include "axon_export.h" // Include the generated export header

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

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT Storage* smalloc(float* data, int size, Device device);
    AXON_EXPORT void sfree(Storage* s, Device device);

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

    AXON_EXPORT Tensor* tmalloc(int* shape, int ndim, Device device, bool requires_grad);
    AXON_EXPORT void tfree(Tensor* t);

    AXON_EXPORT void gmalloc(Tensor* t, float init);
    AXON_EXPORT void gfree(Tensor* t);

    AXON_EXPORT int numel(const int* shape, int ndim);
    AXON_EXPORT int* compute_strides(const int* shape, int ndim);
    AXON_EXPORT bool is_contiguous(Tensor* t);
    AXON_EXPORT bool shapes_equal(const int* shape1, int ndim1, const int* shape2, int ndim2);

#ifdef __cplusplus
}
#endif
#endif

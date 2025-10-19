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
} DeviceType;

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

    typedef struct
    {
        DeviceType type;
        int index;
    } Device;

    AXON_EXPORT Device* dmalloc(DeviceType type, int index);
    AXON_EXPORT void dfree(Device* device);

    AXON_EXPORT Storage* smalloc(float* data, int size, Device* device);
    AXON_EXPORT void sfree(Storage* s, Device* device);

    typedef struct sTensor
    {
        Storage* data;
        struct sTensor* grad;
        int* shape;
        int* strides;
        Device* device;
        int ndim;
        bool requires_grad;
    } Tensor;
    AXON_EXPORT Tensor* tmalloc(int* shape, int ndim, Device* device, bool requires_grad);
    AXON_EXPORT void tfree(Tensor* t);

    AXON_EXPORT void gmalloc(Tensor* t, float init);

    AXON_EXPORT int numel(const int* shape, int ndim);
    AXON_EXPORT int* compute_strides(const int* shape, int ndim);
    AXON_EXPORT bool is_contiguous(Tensor* t);
    AXON_EXPORT bool shapes_equal(const int* shape1, int ndim1, const int* shape2, int ndim2);
    AXON_EXPORT void copy_storage_to_host(Storage* s, Device* device, int size, float* host_buffer);

#ifdef __cplusplus
}
#endif
#endif

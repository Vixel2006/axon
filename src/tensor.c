#include "tensor.h"
#include "logger.h"
#include <cuda_runtime.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool is_contiguous(Tensor* t)
{
    LOG_INFO("is_contiguous: Entering function");
    int expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; --i)
    {
        if (t->shape[i] > 1)
        {
            if (t->strides[i] != expected_stride)
            {
                LOG_INFO("Tensor not contiguous at dim %d (stride=%d, expected=%d)", i,
                         t->strides[i], expected_stride);
                return false;
            }
            expected_stride *= t->shape[i];
        }
    }
    LOG_INFO("Tensor is contiguous");
    return true;
}

bool shapes_equal(const int* shape1, int ndim1, const int* shape2, int ndim2)
{
    LOG_INFO("shapes_equal: Entering function with ndim1=%d, ndim2=%d", ndim1, ndim2);
    if (ndim1 != ndim2) return false;
    for (int i = 0; i < ndim1; ++i)
    {
        if (shape1[i] != shape2[i]) return false;
    }
    return true;
}

int numel(const int* shape, int ndim)
{
    LOG_INFO("numel: Entering function with ndim=%d", ndim);
    if (ndim == 0) return 1;
    if (ndim < 0 || !shape) return 0;

    int size = 1;
    for (int i = 0; i < ndim; ++i)
    {
        if (shape[i] <= 0)
        {
            LOG_WARN("Invalid shape[%d] = %d, returning 0", i, shape[i]);
            return 0;
        }
        size *= shape[i];
    }
    LOG_INFO("numel computed: %d", size);
    return size;
}

int* compute_strides(const int* shape, int ndim)
{
    LOG_INFO("compute_strides: Entering function with ndim=%d", ndim);
    if (ndim <= 0 || !shape) return NULL;

    int* strides = malloc(ndim * sizeof(int));
    if (!strides)
    {
        LOG_ERROR("Failed to allocate strides for ndim=%d", ndim);
        return NULL;
    }

    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    LOG_INFO("Strides allocated at %p for ndim=%d", strides, ndim);
    return strides;
}

Device* dmalloc(DeviceType type, int index)
{
    LOG_INFO("dmalloc: Entering function with type=%d, index=%d", type, index);
    LOG_INFO("Initializing device %d, %d", type, index);
    Device* device = malloc(sizeof(Device));
    if (!device)
    {
        LOG_ERROR("Failed to allocate Device");
        return NULL;
    }
    device->type = type;
    device->index = index;
    device->counter = 1;

    LOG_INFO("Device initialized successfully.");

    return device;
}

void dfree(Device* device)
{
    LOG_INFO("dfree: Entering function");
    if (!device) return;

    if (--device->counter <= 0)
    {
        LOG_INFO("Freeing Device struct at %p with type %d and index %d", device, device->type,
                 device->index);
        free(device);
    }
}

Storage* smalloc(float* data, int size, Device* device)
{
    LOG_INFO("smalloc: Entering function with size=%d", size);
    if (!device)
    {
        LOG_ERROR("smalloc requires a non-NULL device.");
        return NULL;
    }

    if (device->type == CPU)
    {
        return smalloc_cpu(data, size, device);
    }
    else if (device->type == CUDA)
    {
        return smalloc_cuda(data, size, device);
    }
    else
    {
        LOG_ERROR("Unsupported device type for smalloc.");
        return NULL;
    }
}

void sfree(Storage* s, Device* device)
{
    LOG_INFO("sfree: Entering function");
    if (!s) return;

    if (device->type == CPU)
    {
        sfree_cpu(s, device);
    }
    else if (device->type == CUDA)
    {
        sfree_cuda(s, device);
    }
    else
    {
        LOG_ERROR("Unsupported device type for sfree.");
    }
}

void to(Tensor* t, Device* device)
{
    LOG_INFO("to: Entering function");
    LOG_INFO("to() called: current device type %d, target device type %d", t->device->type,
             device->type);

    if (t->device->type == device->type && t->device->index == device->index)
    {
        return;
    }

    Device* old_device = t->device;
    Storage* dbuffer = NULL;

    if (device->type == CUDA)
    {
        dbuffer = smalloc_cuda(t->data->data, t->data->size, device);
    }
    else
    {
        dbuffer = smalloc_cpu(t->data->data, t->data->size, device);
    }

    if (t->data)
    {
        if (old_device->type == CPU)
        {
            sfree_cpu(t->data, old_device);
        }
        else
        {
            sfree_cuda(t->data, old_device);
        }
    }

    t->data = dbuffer;
    t->device = device;
    device->counter++;
    dfree(old_device);

    if (t->requires_grad && t->grad && t->grad->data)
    {
        Storage* gbuffer = NULL;
        if (device->type == CUDA)
        {
            gbuffer = smalloc_cuda(t->grad->data->data, t->grad->data->size, device);
        }
        else
        {
            gbuffer = smalloc_cpu(t->grad->data->data, t->grad->data->size, device);
        }
        if (old_device->type == CPU)
        {
            sfree_cpu(t->grad->data, old_device);
        }
        else
        {
            sfree_cuda(t->grad->data, old_device);
        }
        Device* old_grad_device = t->grad->device;
        t->grad->data = gbuffer;
        t->grad->device = device;
        device->counter++;
        dfree(old_grad_device);
    }
}

void gmalloc(Tensor* t, float init)
{
    LOG_INFO("gmalloc: Entering function with init=%.2f", init);
    if (!t->device)
    {
        LOG_ERROR("gmalloc cannot proceed on a tensor with a NULL device.");
        return;
    }
    int size = numel(t->shape, t->ndim);

    if (!t->grad)
    {
        t->grad = tmalloc(t->shape, t->ndim, t->device, false);
        if (!t->grad)
        {
            LOG_ERROR("Failed to allocate Tensor for grad");
            return;
        }

        t->grad->data = smalloc(NULL, size, t->device);
        if (!t->grad->data)
        {
            LOG_ERROR("Failed to allocate Storage for grad data");
            tfree(t->grad);
            t->grad = NULL;
            return;
        }
        LOG_INFO("Tensor allocated for grad at %p (data=%p) with size=%d", t->grad,
                 t->grad->data->data, size);
    }

    if (t->device->type == CPU)
    {
        for (int i = 0; i < size; ++i)
        {
            t->grad->data->data[i] = init;
        }
    }
    else if (t->device->type == CUDA)
    {
        float* host_init_buffer = (float*) malloc(size * sizeof(float));
        if (!host_init_buffer)
        {
            LOG_ERROR("Failed to allocate host_init_buffer for grad initialization.");
            return;
        }
        for (int i = 0; i < size; ++i)
        {
            host_init_buffer[i] = init;
        }
        cudaError_t err = cudaMemcpy(t->grad->data->data, host_init_buffer, size * sizeof(float),
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            LOG_ERROR("Failed to copy init value to grad on cuda device: %s",
                      cudaGetErrorString(err));
        }
        free(host_init_buffer);
    }
}

Tensor* tmalloc(int* shape, int ndim, Device* device, bool requires_grad)
{
    LOG_INFO("tmalloc: Entering function with ndim=%d, requires_grad=%d", ndim, requires_grad);
    Tensor* t = malloc(sizeof(Tensor));
    if (!t)
    {
        LOG_ERROR("Failed to allocate Tensor");
        return NULL;
    }

    t->ndim = ndim;
    t->shape = malloc(sizeof(int) * ndim);
    if (!t->shape)
    {
        LOG_ERROR("Failed to allocate Tensor shape");
        free(t);
        return NULL;
    }

    if (ndim > 0 && !shape)
    {
        LOG_ERROR("Shape cannot be NULL for ndim > 0 in tmalloc");
        free(t->shape);
        free(t);
        return NULL;
    }
    for (int i = 0; i < t->ndim; ++i)
        t->shape[i] = shape[i];

    t->strides = compute_strides(t->shape, t->ndim);
    t->device = device;
    if (device)
    {
        device->counter++;
    }
    t->requires_grad = requires_grad;
    t->data = NULL;
    t->grad = NULL;

    LOG_INFO("Tensor allocated at %p (ndim=%d, device=%p, requires_grad=%d)", t, t->ndim,
             (void*) t->device, t->requires_grad);
    return t;
}

void tfree(Tensor* t)
{
    LOG_INFO("tfree: Entering function");
    if (!t) return;

    LOG_INFO("Freeing Tensor at %p", t);

    if (t->grad)
    {
        Tensor* grad = t->grad;
        if (grad->shape)
        {
            free(grad->shape);
            grad->shape = NULL;
        }
        if (grad->strides)
        {
            free(grad->strides);
            grad->strides = NULL;
        }
        if (grad->data)
        {
            sfree(grad->data, grad->device);
            grad->data = NULL;
        }
        // The grad holds a reference, so we decrement the device counter.
        if (grad->device)
        {
            dfree(grad->device);
            grad->device = NULL;
        }
        free(grad);
        t->grad = NULL;
    }

    if (t->shape)
    {
        LOG_INFO("Freed shape at %p", t->shape);
        free(t->shape);
        t->shape = NULL;
    }

    if (t->strides)
    {
        LOG_INFO("Freed strides at %p", t->strides);
        free(t->strides);
        t->strides = NULL;
    }

    if (t->data)
    {
        sfree(t->data, t->device);
        t->data = NULL;
    }

    if (t->device)
    {
        dfree(t->device);
        t->device = NULL;
    }

    free(t);
}

void copy_storage_to_host(Storage* s, Device* device, int size, const int* shape,
                          const int* strides, int ndim, float* host_buffer)
{
    LOG_INFO("copy_storage_to_host: Entering function with size=%d", size);
    if (!s || !host_buffer) return;

    if (device->type == CPU)
    {
        memcpy(host_buffer, s->data, size * sizeof(float));
    }
    else if (device->type == CUDA)
    {
        // Create a dummy Tensor object to use is_contiguous
        Tensor dummy_tensor;
        dummy_tensor.shape = (int*) shape;
        dummy_tensor.strides = (int*) strides;
        dummy_tensor.ndim = ndim;

        if (is_contiguous(&dummy_tensor))
        {
            LOG_INFO("copy_storage_to_host: CUDA tensor is contiguous. Using cudaMemcpy.");
            cudaMemcpy(host_buffer, s->data, size * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else
        {
            LOG_INFO("copy_storage_to_host: CUDA tensor is NON-contiguous. Using "
                     "copy_non_contiguous_cuda_to_host.");
            copy_non_contiguous_cuda_to_host(s, shape, strides, ndim, size, host_buffer);
        }
    }
}

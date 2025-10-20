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
    if (ndim1 != ndim2) return false;
    for (int i = 0; i < ndim1; ++i)
    {
        if (shape1[i] != shape2[i]) return false;
    }
    return true;
}

int numel(const int* shape, int ndim)
{
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
    Device* device = malloc(sizeof(Device));
    if (!device)
    {
        LOG_ERROR("Failed to allocate Device");
        return NULL;
    }
    device->type = type;
    device->index = index;
    return device;
}

void dfree(Device* device)
{
    if (!device) return;

    free(device);
}

Storage* smalloc(float* data, int size, Device* device)
{
    if (!device)
    {
        LOG_ERROR("smalloc requires a non-NULL device.");
        return NULL;
    }

    Storage* s = malloc(sizeof(Storage));

    if (!s)
    {
        LOG_ERROR("Failed to allocate Storage");
        return NULL;
    }

    s->size = size;

    if (device->type == CPU)
    {
        s->data = malloc(sizeof(float) * size);

        if (!s->data)
        {
            LOG_ERROR("Failed to allocate Storage data on cpu.");
            free(s);
            return NULL;
        }

        if (data != NULL)
        {
            for (int i = 0; i < size; ++i)
                s->data[i] = data[i];
        }
        else
        {
            for (int i = 0; i < size; ++i)
                s->data[i] = 0.0f;
        }
    }
    else
    {
        // NOTE: Here we set the device we need to save the tensor on. this way we don't want to set
        // the device for each kernel is cuda will just map kernels to the devices that have its
        // data in it.
        cudaError_t err = cudaSetDevice(device->index);

        if (err != cudaSuccess)
        {
            LOG_ERROR("Failed to find cuda device.");
            free(s);
            return NULL;
        }

        err = cudaMalloc((void**) &s->data, size * sizeof(float));

        if (err != cudaSuccess)
        {
            LOG_ERROR("Failed to allocate Storage data on cuda device.");
            free(s);
            return NULL;
        }

        if (data != NULL)
        {
            cudaMemcpy(s->data, data, size * sizeof(float), cudaMemcpyHostToDevice);
        }
        else
        {
            cudaMemset(s->data, 0, size * sizeof(float));
        }
    }

    s->counter = 1;

    LOG_INFO("Storage allocated at %p with size=%d", s, size);
    return s;
}

void sfree(Storage* s, Device* device)
{
    if (!s) return;
    s->counter--;
    LOG_INFO("sfree called, counter=%d for Storage %p", s->counter, s);
    if (s->counter <= 0)
    {
        if (s->data)
        {
            if (!device)
            {
                LOG_ERROR("Cannot free Storage data with NULL device.");
            }
            else
            {
                LOG_INFO("Storage data freed at %p", s->data);
                if (device->type == CPU)
                {
                    SAFE_FREE(&s->data, free);
                }
                else
                {
                    cudaFree(s->data);
                    s->data = NULL;
                }
            }
        }
        SAFE_FREE(&s, free);
    }
}

void to(Tensor* t, Device* device)
{
    if (t->device->type == CPU && device->type == CUDA)
    {
        int size = t->data->size;
        float* data_buffer = (float*) malloc(sizeof(float) * size);
        if (!data_buffer)
        {
            LOG_ERROR("Failed to allocate data_buffer for device transfer.");
            return;
        }

        memcpy(data_buffer, t->data->data, sizeof(float) * size);
        sfree(t->data, t->device);

        t->data = smalloc(data_buffer, size, device);
        SAFE_FREE(&data_buffer, free);

        if (t->requires_grad)
        {
            float* grad_buffer = (float*) malloc(sizeof(float) * size);
            if (!grad_buffer)
            {
                LOG_ERROR("Failed to allocate grad_buffer for device transfer.");
                return;
            }
            memcpy(grad_buffer, t->grad->data->data, sizeof(float) * size);
            sfree(t->grad->data, t->device);

            t->grad->data = smalloc(grad_buffer, size, device);
            SAFE_FREE(&grad_buffer, free);
        }
        dfree(t->device);
        t->device = dmalloc(device->type, device->index);
    }
    else if (t->device->type == CUDA && device->type == CPU)
    {
        int size = t->data->size;
        float* data_buffer = (float*) malloc(sizeof(float) * size);
        if (!data_buffer)
        {
            LOG_ERROR("Failed to allocate data_buffer for device transfer.");
            return;
        }

        cudaMemcpy(data_buffer, t->data->data, sizeof(float) * size, cudaMemcpyDeviceToHost);
        sfree(t->data, t->device);

        t->data = smalloc(data_buffer, size, device);
        SAFE_FREE(&data_buffer, free);

        if (t->requires_grad)
        {
            float* grad_buffer = (float*) malloc(sizeof(float) * size);
            if (!grad_buffer)
            {
                LOG_ERROR("Failed to allocate grad_buffer for device transfer.");
                return;
            }
            cudaMemcpy(grad_buffer, t->grad->data->data, sizeof(float) * size,
                       cudaMemcpyDeviceToHost);
            sfree(t->grad->data, t->device);

            t->grad->data = smalloc(grad_buffer, size, device);
            SAFE_FREE(&grad_buffer, free);
        }
        dfree(t->device);
        t->device = dmalloc(device->type, device->index);
    }
    else if (t->device->type == CUDA && device->type == CUDA)
    {
        if (t->device->index == device->index)
        {
            LOG_INFO("Attempted to transfer tensor to the same CUDA device. No action taken.");
            return;
        }

        int size = t->data->size;

        // NOTE: here we use the cudaMemcpyPeer that copies the data between the two devices
        // directly and if there is any problem we fallback to the slow path that copy from device
        // to host then from host to device
        Storage* storage = smalloc(NULL, size, device);

        cudaError_t err = cudaMemcpyPeer(storage->data, device->index, t->data->data,
                                         t->device->index, sizeof(float) * size);

        sfree(t->data, t->device);

        t->data = storage;

        if (t->requires_grad)
        {
            Storage* grad_storage = smalloc(NULL, size, device);
            err = cudaMemcpyPeer(grad_storage->data, device->index, t->grad->data->data,
                                 t->device->index, sizeof(float) * size);

            sfree(t->data, t->device);

            t->grad->data = grad_storage;
        }

        if (err != cudaSuccess)
        {

            float* data_buffer = (float*) malloc(sizeof(float) * size);
            if (!data_buffer)
            {
                LOG_ERROR("Failed to allocate data_buffer for CUDA to CUDA transfer.");
                return;
            }

            err = cudaMemcpy(data_buffer, t->data->data, sizeof(float) * size,
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                LOG_ERROR("Failed to copy data from source CUDA device to host: %s",
                          cudaGetErrorString(err));
                SAFE_FREE(&data_buffer, free);
                return;
            }

            sfree(t->data, t->device);

            t->data = smalloc(data_buffer, size, device);
            SAFE_FREE(&data_buffer, free);

            if (t->requires_grad)
            {
                float* grad_buffer = (float*) malloc(sizeof(float) * size);
                if (!grad_buffer)
                {
                    LOG_ERROR("Failed to allocate grad_buffer for CUDA to CUDA transfer.");
                    return;
                }
                err = cudaMemcpy(grad_buffer, t->grad->data->data, sizeof(float) * size,
                                 cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    LOG_ERROR("Failed to copy grad from source CUDA device to host: %s",
                              cudaGetErrorString(err));
                    SAFE_FREE(&grad_buffer, free);
                    return;
                }
                sfree(t->grad->data, t->device);

                t->grad->data = smalloc(grad_buffer, size, device);
                SAFE_FREE(&grad_buffer, free);
            }
            dfree(t->device);
        }
        t->device = dmalloc(device->type, device->index);
    }
    else
    {
        LOG_WARN("Attempted to transfer tensor to the same device type or unsupported "
                 "transfer. No action taken.");
    }
}

void gmalloc(Tensor* t, float init)
{
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
    else
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
        SAFE_FREE(&t, free);
        return NULL;
    }

    if (ndim > 0 && !shape)
    {
        LOG_ERROR("Shape cannot be NULL for ndim > 0 in tmalloc");
        SAFE_FREE(&t->shape, free);
        SAFE_FREE(&t, free);
        return NULL;
    }
    for (int i = 0; i < t->ndim; ++i)
        t->shape[i] = shape[i];

    t->strides = compute_strides(t->shape, t->ndim);
    LOG_INFO("tmalloc: t=%p, t->shape=%p, t->strides=%p", t, t->shape, t->strides);
    t->device = device;
    t->requires_grad = requires_grad;
    t->data = NULL;
    t->grad = NULL;

    LOG_INFO("Tensor allocated at %p (ndim=%d, device=%p, requires_grad=%d)", t, t->ndim,
             (void*) t->device, t->requires_grad);
    return t;
}

void tfree(Tensor* t)
{
    if (!t) return;

    LOG_INFO("Freeing Tensor at %p", t);

    if (t->shape)
    {
        LOG_INFO("Freed shape at %p", t->shape);
        SAFE_FREE(&t->shape, free);
    }

    if (t->strides)
    {
        LOG_INFO("Freed strides at %p", t->strides);
        SAFE_FREE(&t->strides, free);
    }

    if (t->data)
    {
        sfree(t->data, t->device);
        t->data = NULL;
    }

    if (t->grad)
    {
        tfree(t->grad);
        t->grad = NULL;
    }

    SAFE_FREE(&t, free);
}

void copy_storage_to_host(Storage* s, Device* device, int size, float* host_buffer)
{
    if (!s || !host_buffer) return;

    if (device->type == CPU)
    {
        memcpy(host_buffer, s->data, size * sizeof(float));
    }
    else
    {
        cudaMemcpy(host_buffer, s->data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

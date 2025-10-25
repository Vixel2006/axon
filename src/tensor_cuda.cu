#include "logger.h"
#include "tensor.h"
#include "tensor_cuda.h"
#include <cuda_runtime.h>

Storage* smalloc_cuda(float* data, int size, Device* device)
{
    if (!device)
    {
        LOG_ERROR("smalloc_cuda requires a non-NULL device.");
        return NULL;
    }

    Storage* s = (Storage*) malloc(sizeof(Storage));

    if (!s)
    {
        LOG_ERROR("Failed to allocate Storage");
        return NULL;
    }

    s->size = size;

    cudaError_t err = cudaSetDevice(device->index);
    if (err != cudaSuccess)
    {
        LOG_ERROR("smalloc_cuda: Failed to set CUDA device %d: %s", device->index,
                  cudaGetErrorString(err));
        free(s);
        return NULL;
    }

    err = cudaMalloc((void**) &s->data, size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("smalloc_cuda: Failed to allocate Storage data on CUDA device %d: %s",
                  device->index, cudaGetErrorString(err));
        free(s);
        return NULL;
    }

    if (data != NULL)
    {
        err = cudaMemcpy(s->data, data, size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            LOG_ERROR("smalloc_cuda: Failed to copy data to CUDA device %d: %s", device->index,
                      cudaGetErrorString(err));
            cudaFree(s->data);
            free(s);
            return NULL;
        }
    }
    else
    {
        err = cudaMemset(s->data, 0, size * sizeof(float));
        if (err != cudaSuccess)
        {
            LOG_ERROR("smalloc_cuda: Failed to memset CUDA device %d: %s", device->index,
                      cudaGetErrorString(err));
            cudaFree(s->data);
            free(s);
            return NULL;
        }
    }

    s->counter = 1;

    LOG_INFO("Storage allocated at %p with size=%d", s, size);
    return s;
}

void sfree_cuda(Storage* s, Device* device)
{
    if (!s) return;
    s->counter--;
    LOG_INFO("sfree_cuda called, counter=%d for Storage %p", s->counter, s);
    if (s->counter <= 0)
    {
        if (s->data)
        {
            LOG_INFO("Storage data freed at %p", s->data);
            cudaError_t err = cudaFree(s->data);
            if (err != cudaSuccess)
            {
                LOG_ERROR("sfree_cuda: Failed to free CUDA memory: %s", cudaGetErrorString(err));
            }
            s->data = NULL;
        }
        SAFE_FREE(&s, free);
    }
}

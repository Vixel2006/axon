#include "logger.h"
#include "tensor.h"
#include "tensor_cuda.h"
#include <assert.h>
#include <cuda_runtime.h>

#define MAX_NDIM 8 // Define MAX_NDIM

// CUDA kernel to copy non-contiguous data from device to a contiguous device buffer
__global__ void copy_non_contiguous_kernel(const float* d_data, const int* shape,
                                           const int* strides, int ndim, float* d_out_buffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Assuming d_out_buffer is linear and contiguous
    // Calculate logical coordinates from linear index
    int logical_coords[MAX_NDIM];
    int temp_idx = idx;
    for (int i = ndim - 1; i >= 0; --i)
    {
        logical_coords[i] = temp_idx % shape[i];
        temp_idx /= shape[i];
    }

    // Calculate memory offset in the non-contiguous d_data
    int memory_offset_elements = 0;
    for (int i = 0; i < ndim; ++i)
    {
        memory_offset_elements += logical_coords[i] * strides[i];
    }
    d_out_buffer[idx] = d_data[memory_offset_elements];
}

// Host-side function to manage the non-contiguous copy from device to host
void copy_non_contiguous_cuda_to_host(Storage* s, const int* shape, const int* strides, int ndim,
                                      int num_elements, float* host_buffer)
{
    LOG_INFO("copy_non_contiguous_cuda_to_host: shape=(%d, %d), strides=(%d, %d), ndim=%d, "
             "num_elements=%d",
             shape[0], shape[1], strides[0], strides[1], ndim, num_elements);
    // Allocate a temporary contiguous device buffer
    float* d_staging_buffer;
    cudaMalloc((void**) &d_staging_buffer, num_elements * sizeof(float));

    // Configure kernel launch parameters
    int blockSize = 256;
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    // Launch kernel to copy from non-contiguous device memory to contiguous device staging buffer
    copy_non_contiguous_kernel<<<numBlocks, blockSize>>>(s->data, shape, strides, ndim,
                                                         d_staging_buffer);
    cudaDeviceSynchronize(); // Wait for kernel to complete

    // Copy from contiguous device staging buffer to host buffer
    cudaMemcpy(host_buffer, d_staging_buffer, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free temporary device buffer
    cudaFree(d_staging_buffer);
}

Storage* smalloc_cuda(float* data, int size, Device* device, Device* src_device)
{
    LOG_INFO("smalloc_cuda: Entering function with size=%d", size);

    if (!device)
    {
        LOG_ERROR("smalloc_cuda requires a non-NULL device.");
        return NULL;
    }

    Storage* s = (Storage*) malloc(sizeof(Storage));

    if (!s)
    {
        LOG_ERROR("Failed to allocate Storage");
        assert(0 && "Failed to allocate Storage");
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
        assert(0 && "smalloc_cuda: Failed to allocate Storage data on CUDA device");
    }

    if (data != NULL)
    {
        if (src_device->type == CPU)
        {
            err = cudaMemcpy(s->data, data, size * sizeof(float), cudaMemcpyHostToDevice);
        }
        else // CUDA to CUDA copy (or device to device)
        {
            err = cudaMemcpy(s->data, data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        if (err != cudaSuccess)
        {
            LOG_ERROR("smalloc_cuda: Failed to copy data to CUDA device %d: %s", device->index,
                      cudaGetErrorString(err));
            cudaFree(s->data);
            free(s);
            assert(0 && "smalloc_cuda: Failed to copy data to CUDA device");
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
            assert(0 && "smalloc_cuda: Failed to memset CUDA device");
        }
    }

    s->counter = 1;

    LOG_INFO("Storage allocated at %p with size=%d", s, size);
    return s;
}

void sfree_cuda(Storage* s, Device* device)
{
    LOG_INFO("sfree_cuda: Entering function");
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

#include "logger.h"
#include "ops/init_ops.h"
#include "ops/unary_ops.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#define CHECK_CUDA()                                                                               \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("CUDA runtime error at %s:%d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
            return;                                                                                \
        }                                                                                          \
    } while (0)

__global__ void relu_kernel(const float* a, float* b, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        // NOTE: Here we use the fmaxf from the cuda math library so we can metigate the diversion
        // that can happens in the warp calcuation if we used b[i] = a[i] > 0 ? a[i] : 0;
        b[i] = fmaxf(a[i], 0);
    }
}

__global__ void log_kernel(const float* a, float* b, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        b[i] = logf(a[i]);
    }
}

__global__ void exp_kernel(const float* a, float* b, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        b[i] = expf(a[i]);
    }
}

__global__ void neg_kernel(const float* a, float* b, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        b[i] = -a[i];
    }
}

__global__ void abs_kernel(const float* a, float* b, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        b[i] = fabsf(a[i]);
    }
}

__global__ void clip_kernel(const float* a, float* b, const float min_val, const float max_val,
                            int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        // NOTE: Here we will use a mix of fminf, fmaxf to metigate the warp diversion problem that
        // will happen if use branching
        b[i] = fmaxf(min_val, fminf(max_val, a[i]));
    }
}

void relu_op_cuda(Tensor* in, Tensor* out)
{
    LOG_INFO("ReLU operation on CUDA running......");
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    float* d_data;
    cudaMalloc((void**) &d_data, sizeof(float) * N);

    relu_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, d_data, N);

    CHECK_CUDA();

    float* h_data;
    cudaMallocHost((void**) &h_data, sizeof(float) * N);
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    from_data(out, h_data);

    SAFE_FREE(&h_data, cudaFreeHost);
    SAFE_FREE(&d_data, cudaFree);

    LOG_INFO("ReLU operation done on CUDA successfully.");
}

void log_op_cuda(Tensor* in, Tensor* out)
{
    LOG_INFO("LOG operation on CUDA running.......");
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    log_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data, N);

    CHECK_CUDA();

    LOG_INFO("LOG operation done on CUDA successfully.");
}

void exp_op_cuda(Tensor* in, Tensor* out)
{
    LOG_INFO("EXP operation on CUDA running.......");
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    exp_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data, N);

    CHECK_CUDA();

    LOG_INFO("EXP operation done on CUDA successfully.");
}

void neg_op_cuda(Tensor* in, Tensor* out)
{
    LOG_INFO("NEG operation on CUDA running.......");
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    neg_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data, N);

    CHECK_CUDA();

    LOG_INFO("NEG operation done on CUDA successfully.");
}

void abs_op_cuda(Tensor* in, Tensor* out)
{
    LOG_INFO("ABS operation on CUDA running.......");
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    abs_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data, N);

    CHECK_CUDA();

    LOG_INFO("ABS operation done on CUDA successfully.");
}

void clip_op_cuda(Tensor* in, Tensor* out, float min_val, float max_val)
{
    LOG_INFO("CLIP operation on CUDA running.......");
    int N = numel(in->shape, in->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    clip_kernel<<<num_blocks, num_threads_per_block>>>(in->data->data, out->data->data, min_val,
                                                       max_val, N);

    CHECK_CUDA();

    LOG_INFO("CLIP operation done on CUDA successfully.");
}

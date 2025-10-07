#include "logger.h"
#include "ops/binary_ops.h"
#include "ops/init_ops.h"
#include <cuda_runtime.h>

#define TILE_DIM 16

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

__global__ void add_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] + b[i];
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] - b[i];
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] * b[i];
    }
}

__global__ void div_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = a[i] / b[i];
    }
}

__global__ void pow_kernel(const float* a, const float* b, float* out, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        out[i] = powf(a[i], b[i]);
    }
}

__global__ void matmul_kernel(const float* a, const float* b, float* out, const int N, const int M,
                              const int K)
{
    int batch = blockIdx.z;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float a_tile[TILE_DIM][TILE_DIM];
    __shared__ float b_tile[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    const float* a_batch = a + batch * N * K;
    const float* b_batch = b + batch * K * M;
    float* c_batch = out + batch * N * M;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) // Corrected outer loop
    {
        int tiledColA = t * TILE_DIM + threadIdx.x;
        int tiledRowB = t * TILE_DIM + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] =
            (row < N && tiledColA < K) ? a_batch[row * K + tiledColA] : 0.0f;

        b_tile[threadIdx.y][threadIdx.x] =
            (tiledRowB < K && col < M) ? b_batch[tiledRowB * M + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) // Corrected inner loop
        {
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (col < M && row < N) c_batch[row * M + col] = sum;
}

void add_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Add kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    float* h_out;
    float* d_out;

    cudaMallocHost((void**) &h_out, sizeof(float) * N);
    cudaMalloc((void**) &d_out, sizeof(float) * N);

    add_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, d_out, N);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    from_data(out, h_out);

    SAFE_FREE(&d_out, cudaFree);
    SAFE_FREE(&h_out, cudaFreeHost);

    CHECK_CUDA();

    LOG_INFO("Add kernel done successfully");
}

void sub_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Sub kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    float* h_out;
    float* d_out;

    cudaMallocHost((void**) &h_out, sizeof(float) * N);
    cudaMalloc((void**) &d_out, sizeof(float) * N);

    sub_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, d_out, N);
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    from_data(out, h_out);

    SAFE_FREE(&d_out, cudaFree);
    SAFE_FREE(&h_out, cudaFreeHost);

    CHECK_CUDA();

    LOG_INFO("Sub kernel done successfully");
}

void mul_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Mul kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    float* h_out;
    float* d_out;

    cudaMallocHost((void**) &h_out, sizeof(float) * N);
    cudaMalloc((void**) &d_out, sizeof(float) * N);

    mul_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, d_out, N);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    from_data(out, h_out);

    SAFE_FREE(&d_out, cudaFree);
    SAFE_FREE(&h_out, cudaFreeHost);
    CHECK_CUDA();

    LOG_INFO("Mul kernel done successfully");
}

void div_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Div kernel starts");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    float* h_out;
    float* d_out;

    cudaMallocHost((void**) &h_out, sizeof(float) * N);
    cudaMalloc((void**) &d_out, sizeof(float) * N);

    div_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, d_out, N);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    from_data(out, h_out);

    SAFE_FREE(&d_out, cudaFree);
    SAFE_FREE(&h_out, cudaFreeHost);

    CHECK_CUDA();

    LOG_INFO("Div kernel done successfully");
}

void pow_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    LOG_INFO("Pow kernel starts......");
    int N = numel(a->shape, a->ndim);

    int num_threads_per_block = 256;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    float* h_out;
    float* d_out;

    cudaMallocHost((void**) &h_out, sizeof(float) * N);
    cudaMalloc((void**) &d_out, sizeof(float) * N);

    pow_kernel<<<num_blocks, num_threads_per_block>>>(a->data->data, b->data->data, d_out, N);

    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    from_data(out, h_out);

    SAFE_FREE(&d_out, cudaFree);
    SAFE_FREE(&h_out, cudaFreeHost);

    CHECK_CUDA();

    LOG_INFO("Pow kernel done successfully");
}

void matmul_op_cuda(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P)
{
    LOG_INFO("MATMUL kernel starts......");

    int B = 1;
    for (int i = 0; i < a->ndim - 2; ++i)
    {
        B += a->shape[i];
    }

    float* h_out;
    float* d_out;

    cudaMallocHost((void**) &h_out, sizeof(float) * B * N * P);
    cudaMalloc((void**) &d_out, sizeof(float) * B * N * P);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((P + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM, B);

    matmul_kernel<<<grid, block>>>(a->data->data, b->data->data, d_out, N, P, K);

    cudaMemcpy(h_out, d_out, sizeof(float) * N * B * P, cudaMemcpyDeviceToHost);

    from_data(out, h_out);

    SAFE_FREE(&d_out, cudaFree);
    SAFE_FREE(&h_out, cudaFreeHost);

    CHECK_CUDA();

    LOG_INFO("MATMUL kernel done successfully");
}

void conv2d_op_cuda(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size,
                    const int* stride, const int padding)
{
    (void) in;
    (void) kernel;
    (void) out;
    (void) kernel_size;
    (void) stride;
    (void) padding;
    LOG_WARN("conv2d_op_cuda: CUDA implementation not available yet.");
}

void dot_op_cuda(Tensor* a, Tensor* b, Tensor* out)
{
    (void) a;
    (void) b;
    (void) out;
    LOG_WARN("dot_op_cuda: CUDA implementation not available yet.");
}

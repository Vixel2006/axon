#include "ops/cuda/binary.h"
#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)

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

    // NOTE: Here we did (K + TILE_DIM - 1) / TILE_DIM instead of K / TILE_DIM because in the second
    // case if there is leftover that is not divisable by TILE_DIM it will not be calculated
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        int tiledColA = t * TILE_DIM + threadIdx.x;
        int tiledRowB = t * TILE_DIM + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] =
            (row < N && tiledColA < K) ? a_batch[row * K + tiledColA] : 0.0f;

        b_tile[threadIdx.y][threadIdx.x] =
            (tiledRowB < K && col < M) ? b_batch[tiledRowB * M + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
        {
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (col < M && row < N) c_batch[row * M + col] = sum;
}

extern "C" void matmul_op_cuda(Tensor* a, Tensor* b, Tensor* out, int N, int K, int P)
{
    LOG_INFO("matmul_op_cuda: Entering function with N=%d, K=%d, P=%d", N, K, P);
    LOG_INFO("MATMUL kernel starts......");

    int B = 1;
    for (int i = 0; i < a->ndim - 2; ++i)
    {
        B *= a->shape[i];
    }

    out->data = (Storage*) malloc(sizeof(Storage));
    if (!out->data)
    {
        LOG_ERROR("Failed to allocate Storage for out tensor in matmul_op_cuda");
        assert(0 && "Failed to allocate Storage for out tensor in matmul_op_cuda");
    }
    out->data->counter = 1;
    out->data->size = B * N * P;
    cudaError_t err = cudaMalloc((void**) &out->data->data, out->data->size * sizeof(float));
    if (err != cudaSuccess)
    {
        LOG_ERROR("Failed to allocate CUDA memory for out->data->data in matmul_op_cuda: %s",
                  cudaGetErrorString(err));
        SAFE_FREE(&out->data, free);
        assert(0 && "Failed to allocate CUDA memory for out->data->data in matmul_op_cuda");
    }

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((P + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM, B);

    matmul_kernel<<<grid, block>>>(a->data->data, b->data->data, out->data->data, N, P, K);

    CHECK_CUDA();

    LOG_INFO("MATMUL kernel done successfully");
}

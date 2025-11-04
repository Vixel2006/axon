#include "logger.h"
#include "optimizers/optimizers.h"
#include "utils/indexing.cuh"
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(err)                                                                            \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            LOG_ERROR("cuda-runtime error at %s %d: %s", __FILE__, __LINE__,                       \
                      cudaGetErrorString(err));                                                    \
        }                                                                                          \
    } while (0)

__device__ inline float adam_update(float param_grad, float m_estimate, float v_estimate,
                                    float beta1, float beta2, float beta1_pow_t, float beta2_pow_t,
                                    float lr, float epsilon)
{
    m_estimate = beta1 * m_estimate + (1.0f - beta1) * param_grad;
    v_estimate = beta2 * v_estimate + (1.0f - beta2) * (param_grad * param_grad);
    float m_hat = m_estimate / (1.0f - beta1_pow_t);
    float v_hat = v_estimate / (1.0f - beta2_pow_t);
    return lr * m_hat / (sqrtf(v_hat) + epsilon);
}

__global__ void adam_kernel_contig(float* param_data, float* param_grad, float* m_estimates,
                                   float* v_estimates, float lr, float beta1, float beta2,
                                   float beta1_pow_t, float beta2_pow_t, float epsilon, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        float current_param_grad = param_grad[i];
        float m_estimate = m_estimates[i];
        float v_estimate = v_estimates[i];

        m_estimate = beta1 * m_estimate + (1.0f - beta1) * current_param_grad;
        v_estimate =
            beta2 * v_estimate + (1.0f - beta2) * (current_param_grad * current_param_grad);

        float m_hat = m_estimate / (1.0f - beta1_pow_t);
        float v_hat = v_estimate / (1.0f - beta2_pow_t);

        float update_term = lr * m_hat / (sqrtf(v_hat) + epsilon);

        param_data[i] -= update_term;
        m_estimates[i] = m_estimate; // Write back updated m_estimate
        v_estimates[i] = v_estimate; // Write back updated v_estimate
    }
}

__global__ void adam_kernel_noncontig(float* param_data, float* param_grad, float* m_estimates,
                                      float* v_estimates, float lr, float beta1, float beta2,
                                      float beta1_pow_t, float beta2_pow_t, float epsilon,
                                      const int* shape, const int* strides, int ndim, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride)
    {
        int data_idx = get_idx(shape, strides, ndim, i);

        float current_param_grad = param_grad[data_idx];
        float m_estimate = m_estimates[data_idx];
        float v_estimate = v_estimates[data_idx];

        m_estimate = beta1 * m_estimate + (1.0f - beta1) * current_param_grad;
        v_estimate =
            beta2 * v_estimate + (1.0f - beta2) * (current_param_grad * current_param_grad);

        float m_hat = m_estimate / (1.0f - beta1_pow_t);
        float v_hat = v_estimate / (1.0f - beta2_pow_t);

        float update_term = lr * m_hat / (sqrtf(v_hat) + epsilon);

        param_data[data_idx] -= update_term;
        m_estimates[data_idx] = m_estimate; // Write back updated m_estimate
        v_estimates[data_idx] = v_estimate; // Write back updated v_estimate
    }
}

void adam_cuda(Tensor** params, Tensor** m_estimates, Tensor** v_estimates, int num_params,
               int time_step, float learning_rate, float beta1, float beta2, float epsilon)
{
    LOG_INFO("adam_cuda: Entering function with num_params=%d, time_step=%d, learning_rate=%.4f",
             num_params, time_step, learning_rate);

    for (int i = 0; i < num_params; ++i)
    {
        if (!params[i]->grad || !params[i]->grad->data || !params[i]->grad->data->data)
        {
            LOG_WARN("adam_cuda: Parameter %d requires grad but its grad tensor or data is "
                     "NULL. Skipping.",
                     i);
            continue;
        }
        int N = numel(params[i]->shape, params[i]->ndim);

        float* current_m_estimates = m_estimates[i]->data->data;
        float* current_v_estimates = v_estimates[i]->data->data;
        float beta1_pow_t = powf(beta1, time_step);
        float beta2_pow_t = powf(beta2, time_step);

        int num_threads_per_block = 256;
        int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

        if (is_contiguous(params[i]))
        {
            adam_kernel_contig<<<num_blocks, num_threads_per_block>>>(
                params[i]->data->data, params[i]->grad->data->data, current_m_estimates,
                current_v_estimates, learning_rate, beta1, beta2, beta1_pow_t, beta2_pow_t, epsilon,
                N);

            CHECK_CUDA(cudaGetLastError());
        }
        else
        {
            adam_kernel_noncontig<<<num_blocks, num_threads_per_block>>>(
                params[i]->data->data, params[i]->grad->data->data, current_m_estimates,
                current_v_estimates, learning_rate, beta1, beta2, beta1_pow_t, beta2_pow_t, epsilon,
                params[i]->shape, params[i]->strides, params[i]->ndim, N);

            CHECK_CUDA(cudaGetLastError());
        }
    }

    LOG_INFO("adam_cuda: Adam optimization on cuda complete.");
}

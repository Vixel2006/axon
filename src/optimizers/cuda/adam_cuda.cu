#include "logger.h"
#include "optimizers/optimizers.h"

// TODO: Implement CUDA kernels for adam
void adam_cuda(Tensor** params, Tensor** m_estimates, Tensor** v_estimates, int num_params,
               int time_step, float learning_rate, float beta1, float beta2, float epsilon)
{
    LOG_WARN("adam_cuda: CUDA implementation not available yet.");
}

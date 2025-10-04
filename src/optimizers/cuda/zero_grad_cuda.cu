#include "logger.h"
#include "optimizers/optimizers.h"

// TODO: Implement CUDA kernels for zero_grad
void zero_grad_cuda(Tensor** parameters, int num_parameters)
{
    LOG_WARN("zero_grad_cuda: CUDA implementation not available yet.");
}

#include "logger.h"
#include "optimizers/optimizers.h"

#include "cuda_utils.h"
#include "logger.h"
#include "optimizers/optimizers.h"

void sgd_cuda(Tensor** params, int num_params, float lr)
{
    LOG_INFO("sgd_cuda: Applying SGD optimization (CUDA)");
}

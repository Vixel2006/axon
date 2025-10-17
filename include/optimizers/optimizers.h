#ifndef AXON_OPTIMIZERS_H
#define AXON_OPTIMIZERS_H

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    AXON_EXPORT void sgd_cpu(Tensor** params, int num_params, float lr);
    AXON_EXPORT void adam_cpu(Tensor** params, Tensor** m_estimates, Tensor** v_estimates,
                              int num_params, int time_step, float learning_rate, float beta1,
                              float beta2, float epsilon);
    AXON_EXPORT void zero_grad_cpu(Tensor** parameters, int num_parameters);


    AXON_EXPORT void zero_grad_cuda(Tensor** parameters, int num_parameters);

#ifdef __cplusplus
}
#endif

#endif

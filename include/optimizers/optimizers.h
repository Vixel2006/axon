#ifndef AXON_OPTIMIZERS_H
#define AXON_OPTIMIZERS_H

#include "tensor.h"

void sgd(Tensor** params, int num_params, float lr);
void adam(Tensor** params, Tensor** m_estimates, Tensor** v_estimates, int num_params,
          int time_step, float learning_rate, float beta1, float beta2, float epsilon);
void zero_grad(Tensor** parameters, int num_parameters);

#endif

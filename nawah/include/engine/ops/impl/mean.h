#ifndef NAWAH_OPS_MEAN_H
#define NAWAH_OPS_MEAN_H

#include <stdexcept>
#include <vector>

class Tensor;

Tensor mean_cpu(const Tensor& a, int dim, bool keepdim);
Tensor mean_gpu(const Tensor& a, int dim, bool keepdim);

#endif


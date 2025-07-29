#ifndef NAWAH_OPS_SUM_H
#define NAWAH_OPS_SUM_H

#include <stdexcept>
#include <vector>

class Tensor;

Tensor sum_cpu(const Tensor& a, int dim, bool keepdim);
Tensor sum_gpu(const Tensor& a, int dim, bool keepdim);

#endif

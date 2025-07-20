#ifndef NAWAH_OPS_MUL_H
#define NAWAH_OPS_MUL_H

#include <stdexcept>
#include <vector>

class Tensor;

Tensor mul_cpu(const Tensor &a, float b);
Tensor mul_gpu(const Tensor &a, float b);

#endif

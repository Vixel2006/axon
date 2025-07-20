#ifndef NAWAH_OPS_ADD_H
#define NAWAH_OPS_ADD_H

#include <immintrin.h>
#include <stdexcept>
#include <vector>

class Tensor;

Tensor add_cpu(const Tensor &a, const Tensor &b);
Tensor add_gpu(const Tensor &a, const Tensor &b);
#endif

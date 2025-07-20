#ifndef NAWAH_OPS_MUL_H
#define NAWAH_OPS_MUL_H

#include <immintrin.h>
#include <stdexcept>
#include <vector>

class Tensor;

Tensor mul_cpu(const Tensor &a, float b);
#endif

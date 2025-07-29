#ifndef NAWAH_OPS_CONV_H
#define NAWAH_OPS_CONV_H

#include <stdexcept>
#include <vector>
#include "engine/ops/traits/ops_trait.h"

class Tensor;

Tensor conv_cpu(const Tensor& a, const Tensor& b);
Tensor conv_gpu(const Tensor& a, const Tensor& b);

#endif

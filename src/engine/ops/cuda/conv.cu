#include "engine/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include "autograd/ops.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>


Tensor CudaOps::conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) {
}


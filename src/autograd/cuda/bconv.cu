#include "autograd/ops.h"
#include "tensor.h"
#include "helpers.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <vector>

#include <cublas_v2.h>


void CudaAutograd::conv2d(Tensor& out, std::vector<Tensor>& prev, int stride, int padding) {
}

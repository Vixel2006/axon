#include "ops/cuda/binary.h"
#include "ops/cuda/init.h" // For smalloc, gmalloc (if needed)

extern "C" void conv2d_op_cuda(Tensor* in, Tensor* kernel, Tensor* out, const int* kernel_size,
                               const int* stride, const int padding)
{
    LOG_INFO("conv2d_op_cuda: Entering function");
    (void) in;
    (void) kernel;
    (void) out;
    (void) kernel_size;
    (void) stride;
    (void) padding;
    LOG_WARN("conv2d_op_cuda: CUDA implementation not available yet.");
}

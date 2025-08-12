#include "tensor.h"
#include "helpers.h"
#include "engine/ops.h"
#include "autograd/ops.h"

Tensor CudaOps::flatten(Tensor& t) {
  int64_t batch_size = t.shape()[0];

  int64_t flattened = 1;

  for (int i = 1 ; i < t.shape().size(); ++i) {
    flattened *= t.shape()[i];
  }

  std::vector<int64_t> new_shape = {batch_size, flattened};
  std::vector<int64_t> new_strides = compute_strides_(new_shape);

  std::shared_ptr<void> data = t.data_ptr();
  Tensor out = Tensor(new_shape, new_strides, t.dtype(), t.device(), data, 0, t.requires_grad(), nullptr, std::nullopt);

  if (t.requires_grad()) {
    out.set_ctx({t}, CudaAutograd::flatten);
  }

  return t;
}

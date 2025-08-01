#include "autograd/badd.h"
#include "tensor.h"

void backward_add_cpu(const Tensor& out, std::vector<Tensor>& prev) {
  if (prev[0].requires_grad()) {
    prev[0].grad() += out.grad();
  }

  if (prev[1].requires_grad()) {
    prev[1].grad() += out.grad();
  }
}


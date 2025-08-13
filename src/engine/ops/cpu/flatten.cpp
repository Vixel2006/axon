#include "tensor.h"
#include "helpers.h"
#include "engine/ops.h"
#include "autograd/ops.h"
#include <stdexcept>

Tensor CpuOps::flatten(Tensor& t) {
    if (!t.is_contiguous()) {
        throw std::runtime_error("flatten(): can only be called on a contiguous tensor.");
    }

    if (t.ndim() <= 1) {
        return t; 
    }

    int64_t batch_size = t.shape()[0];
    int64_t flattened_features = 1;

    for (size_t i = 1; i < t.ndim(); ++i) {
        flattened_features *= t.shape()[i];
    }

    std::vector<int64_t> new_shape = {batch_size, flattened_features};

    Tensor out = t.view(new_shape);

    if (t.requires_grad()) {
        auto backward_fn = [](Tensor& grad_output, std::vector<Tensor>& inputs) {
            CpuAutograd::flatten(grad_output, inputs);
        };

        out.set_ctx({t}, backward_fn);
    }

    return out;
}

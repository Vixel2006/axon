#include "ops/ops.h"
#include <stdexcept>
#include "device.h"

Tensor AddTrait::apply(const Tensor& a, const Tensor& b) {
    Device a_device = a.device();
    Device b_device = b.device();

    if (a_device.type != b_device.type) {
        throw std::runtime_error("AddTrait can only be applied to tensors on the same device.");
    }

    if (a_device.type == DeviceType::CPU) {
        return add_cpu(a, b);
    } else if (a_device.type == DeviceType::CUDA) {
        return add_gpu(a, b);
    } else {
        throw std::runtime_error("AddTrait can only be applied to tensors on CPU or CUDA devices.");
    }
}

Tensor SubTrait::apply(const Tensor& a, const Tensor& b) {
    Device a_device = a.device();
    Device b_device = b.device();

    if (a_device.type != b_device.type) {
        throw std::runtime_error("SubTrait can only be applied to tensors on the same device.");
    }

    if (a_device.type == DeviceType::CPU) {
        return sub_cpu(a, b);
    } else if (a_device.type == DeviceType::CUDA) {
        return sub_gpu(a, b);
    } else {
        throw std::runtime_error("SubTrait can only be applied to tensors on CPU or CUDA devices.");
    }
}

/*
Tensor MulTrait::apply(const Tensor& a, const Tensor& b) {
    Device a_device = a.tensor.device();
    Device b_device = b.tensor.device();

    if (a_device.type != b_device.type) {
        throw std::runtime_error("MulTrait can only be applied to tensors on the same device.");
    }

    if (a_device.type == DeviceType::CPU) {
        return mul_cpu(a.tensor, b.tensor);
    } else if (a_device.type == DeviceType::CUDA) {
        return mul_gpu(a.tensor, b.tensor);
    } else {
        throw std::runtime_error("MulTrait can only be applied to tensors on CPU or CUDA devices.");
    }
}
*/

Tensor MatMulTrait::apply(const Tensor& a, const Tensor& b) {
    Device a_device = a.device();
    Device b_device = b.device();

    if (a_device.type != b_device.type) {
        throw std::runtime_error("MatMulTrait can only be applied to tensors on the same device.");
    }

    if (a_device.type == DeviceType::CPU) {
        return matmul_cpu(a, b);
    } else if (a_device.type == DeviceType::CUDA) {
        throw std::runtime_error("MatMulTrait can only be applied to tensors on CPU devices.");
    } else {
        throw std::runtime_error("MatMulTrait can only be applied to tensors on CPU or CUDA devices.");
    }
}

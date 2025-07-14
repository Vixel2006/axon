#include <numeric>
#include <functional>
#include <stdexcept>
#include "tensor.h"
#include "allocator/allocatorFactory.h"
#include "helpers.h"
#include <vector>
#include <cuda_runtime.h>



std::vector<__int64_t> compute_strides_(const std::vector<__int64_t> &shape)
{
    __int64_t acc = 1;
    std::vector<__int64_t> strides(shape.size());

    for (int i = shape.size() - 1; i >= 0; --i)
    {
        strides[i] = acc;
        acc *= shape[i];
    }

    return strides;
}

bool Tensor::is_contiguous() const
{
    if (shape_.empty())
        return true;
    __int64_t acc = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        if (strides_[i] != acc)
        {
            return false;
        }
        acc *= shape_[i];
    }
    return true;
}

Tensor::Tensor(const std::vector<__int64_t> &shape, DType dtype, Device device)
    : shape_(shape),
      strides_(compute_strides_(shape)),
      dtype_(dtype),
      device_(device)
{
    size_t num_elements = this->numel();
    if (num_elements == 0)
    {
        return;
    }

    size_t size_in_bytes = num_elements * DtypeToSize(dtype_);

    auto allocator = AllocatorFactory::get(device_);
    void *raw_ptr = allocator->allocate(size_in_bytes);

    auto deleter = [allocator](void *ptr)
    {
        allocator->deallocate(ptr);
    };

    data_ptr_ = std::shared_ptr<void>(raw_ptr, deleter);
}

Tensor::Tensor(const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, DType dtype, Device device, std::shared_ptr<void> data_ptr) : shape_(shape), strides_(strides), dtype_(dtype), device_(device), data_ptr_(data_ptr)
{
    if (this->strides_.size() != this->shape_.size())
    {
        throw std::runtime_error("Shape and stride dimensions mismatch in Tensor constructor.");
    }
}

size_t Tensor::numel() const
{
    if (this->shape_.empty())
    {
        return 0;
    }

    return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<__int64_t>());
}

Tensor Tensor::view(std::vector<__int64_t> &new_shape) const
{
    __int64_t known_product = 1;
    __int64_t inferred_index = -1;

    for (size_t i = 0; i < new_shape.size(); ++i)
    {
        __int64_t dim = new_shape[i];

        if (dim == -1)
        {
            if (inferred_index != -1)
            {
                throw std::invalid_argument(
                    "view(): only one dimension can be inferred (-1), but got another at index " + std::to_string(i));
            }
            inferred_index = i;
        }
        else if (dim <= 0)
        {
            throw std::invalid_argument(
                "view(): shape dimension at index " + std::to_string(i) + " must be > 0 or -1 for inference, but got " + std::to_string(dim));
        }
        else
        {
            known_product *= dim;
        }
    }

    __int64_t total = this->numel();

    if (inferred_index != -1)
    {
        if (total % known_product != 0)
        {
            throw std::invalid_argument(
                "view(): cannot infer missing dimension at index " + std::to_string(inferred_index) +
                " — product of known dims = " + std::to_string(known_product) +
                " does not divide total elements = " + std::to_string(total));
        }

        new_shape[inferred_index] = total / known_product;
    }

    __int64_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), 1LL, std::multiplies<__int64_t>());
    if (new_numel != total)
    {
        throw std::invalid_argument(
            "view(): mismatch — original numel = " + std::to_string(total) +
            ", new shape produces = " + std::to_string(new_numel));
    }

    if (!this->is_contiguous())
    {
        throw std::runtime_error("view(): tensor must be contiguous to be reshaped.");
    }

    std::vector<__int64_t> new_strides = compute_strides_(new_shape);
    return Tensor(new_shape, new_strides, this->dtype_, this->device_, this->data_ptr_);
}

Tensor Tensor::squeeze(int dim)
{
    if (dim == -1)
        dim = shape_.size() - 1;

    if (dim < -1 || dim >= static_cast<int>(shape_.size()))
    {
        throw std::out_of_range(
            "squeeze(): Dimension " + std::to_string(dim) +
            " is out of bounds for tensor with " + std::to_string(shape_.size()) + " dimensions.");
    }

    if (shape_[dim] != 1)
    {
        throw std::runtime_error(
            "squeeze(): Cannot squeeze dimension " + std::to_string(dim) +
            " with size " + std::to_string(shape_[dim]) +
            ". Only dimensions of size 1 can be squeezed.");
    }

    std::vector<__int64_t> new_shape = shape_;
    std::vector<__int64_t> new_strides = strides_;

    new_shape.erase(new_shape.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);

    return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_);
}

Tensor Tensor::unsqueeze(int dim)
{
    std::vector<__int64_t> new_shape = shape_;

    if (dim == -1)
    {
        new_shape.push_back(1);
    }
    else if (dim < 0 || dim > static_cast<int>(shape_.size()))
    {
        throw std::out_of_range(
            "unsqueeze(): Dimension " + std::to_string(dim) +
            " is out of bounds. Can insert at most at index " +
            std::to_string(shape_.size()) + " for tensor with " +
            std::to_string(shape_.size()) + " dimensions.");
    }
    else
    {
        new_shape.insert(new_shape.begin() + dim, 1);
    }

    std::vector<__int64_t> new_strides = compute_strides_(new_shape);
    return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_);
}

Tensor Tensor::permute(const std::vector<int> &order)
{
    if (order.size() != shape_.size())
    {
        throw std::invalid_argument(
            "permute(): `order` must have the same number of dimensions as tensor shape. "
            "Expected " +
            std::to_string(shape_.size()) + ", got " + std::to_string(order.size()) + ".");
    }

    std::vector<bool> seen(order.size(), false);
    for (int i : order)
    {
        if (i < 0 || i > shape_.size())
        {
            throw std::out_of_range(
                "permute(): each index in `order` must be in range [0, " +
                std::to_string(shape_.size() - 1) + "], but got " + std::to_string(i) + ".");
        }
        if (seen[i])
        {
            throw std::invalid_argument(
                "permute(): duplicate index " + std::to_string(i) + " in `order`.");
        }
        seen[i] = true;
    }

    std::vector<int64_t> new_shape(shape_.size());
    std::vector<int64_t> new_strides(shape_.size());

    for (size_t i = 0; i < order.size(); ++i)
    {
        new_shape[i] = shape_[order[i]];
        new_strides[i] = strides_[order[i]];
    }

    return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_);
}

Tensor Tensor::transpose(int n, int m) const
{
    const size_t rank = shape_.size();

    if (n < 0)
        n += rank;
    if (m < 0)
        m += rank;

    if (n < 0 || n >= rank)
    {
        throw std::out_of_range(
            "transpose(): dimension `n` is out of bounds. Got " + std::to_string(n) +
            ", but tensor has rank " + std::to_string(rank) + ".");
    }
    if (m < 0 || m >= rank)
    {
        throw std::out_of_range(
            "transpose(): dimension `m` is out of bounds. Got " + std::to_string(m) +
            ", but tensor has rank " + std::to_string(rank) + ".");
    }

    if (n == m)
    {
        throw std::invalid_argument(
            "transpose(): dimensions `n` and `m` must be different, but both are " + std::to_string(n) + ".");
    }

    std::vector<__int64_t> new_shape = shape_;
    std::vector<__int64_t> new_strides = strides_;

    std::swap(new_shape[n], new_shape[m]);
    std::swap(new_strides[n], new_strides[m]);

    return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_);
}

Tensor Tensor::expand(const std::vector<__int64_t> &new_shape) const {
    if (new_shape.size() != shape_.size()) {
        throw std::runtime_error(
            "Tensor::expand error: Dimensionality mismatch. "
            "Tried to expand from shape " + shapeToString(shape_) +
            " to shape " + shapeToString(new_shape) + ". "
            "Both shapes must have the same number of dimensions (" +
            std::to_string(shape_.size()) + " expected, got " +
            std::to_string(new_shape.size()) + ")."
        );
    }

    std::vector<__int64_t> new_strides = strides_;

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] != shape_[i] && shape_[i] != 1) {
            throw std::runtime_error(
                "Tensor::expand error: Cannot expand dimension " + std::to_string(i) +
                " from size " + std::to_string(shape_[i]) + " to " +
                std::to_string(new_shape[i]) + ". "
                "Only dimensions of size 1 can be expanded."
            );
        } else if (new_shape[i] != shape_[i]) {
            new_strides[i] = 0;
        }
    }


    return Tensor(new_shape, new_strides, dtype_, device_, data_ptr_);
}

Tensor::~Tensor() {}

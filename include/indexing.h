#pragma once

#ifndef INDEXING_H
#define INDEXING_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm> // For std::clamp

// The base class remains the same
class IndexStrategy
{
public:
    virtual ~IndexStrategy() = default;
    virtual void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const = 0;
};

/**
 * @brief Handles integer indexing (e.g., tensor[5]).
 *
 * This reduces the rank of the tensor by one. It updates the offset
 * but does not add to the new shape or new strides.
 */
class IntegerIndex : public IndexStrategy
{
public:
    explicit IntegerIndex(__int64_t index) : index_(index) {}

    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override
    {
        const __int64_t dim_size = shape[dim];
        __int64_t idx = index_;

        // Step 1: Normalize negative index
        if (idx < 0)
        {
            idx += dim_size;
        }

        // Step 2: Bounds checking
        if (idx < 0 || idx >= dim_size)
        {
            throw std::out_of_range("Index " + std::to_string(index_) + " is out of bounds for dimension " + std::to_string(dim) + " with size " + std::to_string(dim_size));
        }

        // Step 3: Update offset. This "selects" the slice.
        offset += strides[dim] * idx;

        // Step 4: Do NOT add to new_shape or new_strides, as this dimension is removed.
    }

private:
    __int64_t index_;
};

/**
 * @brief Handles a full slice (e.g., tensor[:]).
 *
 * This keeps the dimension, adding its original shape and stride to the new tensor view.
 */
class FullSlice : public IndexStrategy
{
public:
    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override
    {
        // A full slice just copies the existing dimension's info.
        new_shape.push_back(shape[dim]);
        new_strides.push_back(strides[dim]);
        // Offset is not changed because we start at index 0 of this dimension.
    }
};

/**
 * @brief Handles start:end:step slicing (e.g., tensor[2:10:2]).
 *
 * This is the most complex strategy. It correctly normalizes start, end, and step
 * to calculate the new dimension's length, the new stride, and the starting offset.
 * This implementation robustly handles positive and negative steps and edge cases.
 */
class SliceIndex : public IndexStrategy
{
public:
    SliceIndex(__int64_t start, __int64_t end, __int64_t step)
        : start_(start), end_(end), step_(step)
    {
        if (step_ == 0)
        {
            throw std::invalid_argument("slice step cannot be zero");
        }
    }

    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override
    {
        const __int64_t dim_size = shape[dim];
        __int64_t start = start_;
        __int64_t end = end_;
        __int64_t step = step_;

        __int64_t final_start = 0;
        __int64_t final_len = 0;

        if (step > 0)
        {
            // --- Handle Forward Slicing (step > 0) ---
            // Normalize start
            if (start < 0)
                start += dim_size;
            start = std::clamp(start, static_cast<__int64_t>(0), dim_size);

            // Normalize end
            if (end < 0)
                end += dim_size;
            end = std::clamp(end, static_cast<__int64_t>(0), dim_size);

            // Calculate length
            if (end > start)
            {
                final_len = (end - start + step - 1) / step;
            }
            else
            {
                final_len = 0;
            }
            final_start = start;
        }
        else
        { // step < 0
            // --- Handle Backward Slicing (step < 0) ---
            // Normalize start
            if (start < 0)
                start += dim_size;
            start = std::clamp(start, static_cast<__int64_t>(-1), dim_size - 1);

            // Normalize end
            if (end < 0)
                end += dim_size;
            end = std::clamp(end, static_cast<__int64_t>(-1), dim_size - 1);

            // Calculate length
            if (end < start)
            {
                final_len = (start - end + (-step) - 1) / (-step);
            }
            else
            {
                final_len = 0;
            }
            final_start = start;
        }

        // Apply results to the new tensor view
        new_shape.push_back(final_len);
        new_strides.push_back(strides[dim] * step);
        if (final_len > 0)
        {
            offset += strides[dim] * final_start;
        }
    }

private:
    __int64_t start_, end_, step_;
};

class EllipsisIndex : public IndexStrategy {
public:
    // This apply will likely not be called directly if get_item handles it,
    // but we implement it for completeness. It's just a full slice.
    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override {
        new_shape.push_back(shape[dim]);
        new_strides.push_back(strides[dim]);
    }
};

// Strategy for adding a new axis (None)
class NewAxisIndex : public IndexStrategy {
public:
    // This 'apply' is special: it IGNORES the input dimension 'dim'
    // because it's creating a new one from scratch.
    void apply(int dim, const std::vector<__int64_t> &shape, const std::vector<__int64_t> &strides, __int64_t &offset, std::vector<__int64_t> &new_shape, std::vector<__int64_t> &new_strides) const override {
        new_shape.push_back(1);
        new_strides.push_back(0); // A dimension of size 1 with stride 0 can be broadcast
    }
};


#endif
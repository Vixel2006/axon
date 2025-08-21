#pragma once

#ifndef INDEXING_H
#define INDEXING_H

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

class IndexStrategy {
public:
  virtual ~IndexStrategy() = default;
  virtual void apply(int dim, const std::vector<__int64_t> &shape,
                     const std::vector<__int64_t> &strides, __int64_t &offset,
                     std::vector<__int64_t> &new_shape,
                     std::vector<__int64_t> &new_strides) const = 0;
};

class IntegerIndex : public IndexStrategy {
public:
  explicit IntegerIndex(__int64_t index) : index_(index) {}

  void apply(int dim, const std::vector<__int64_t> &shape,
             const std::vector<__int64_t> &strides, __int64_t &offset,
             std::vector<__int64_t> &new_shape,
             std::vector<__int64_t> &new_strides) const override {
    const __int64_t dim_size = shape[dim];
    __int64_t idx = index_;

    if (idx < 0) {
      idx += dim_size;
    }

    if (idx < 0 || idx >= dim_size) {
      throw std::out_of_range("Index " + std::to_string(index_) +
                              " is out of bounds for dimension " +
                              std::to_string(dim) + " with size " +
                              std::to_string(dim_size));
    }

    offset += strides[dim] * idx;
  }

private:
  __int64_t index_;
};

class FullSlice : public IndexStrategy {
public:
  void apply(int dim, const std::vector<__int64_t> &shape,
             const std::vector<__int64_t> &strides, __int64_t &offset,
             std::vector<__int64_t> &new_shape,
             std::vector<__int64_t> &new_strides) const override {
    new_shape.push_back(shape[dim]);
    new_strides.push_back(strides[dim]);
  }
};

class SliceIndex : public IndexStrategy {
public:
  SliceIndex(__int64_t start, __int64_t end, __int64_t step)
      : start_(start), end_(end), step_(step) {
    if (step_ == 0) {
      throw std::invalid_argument("slice step cannot be zero");
    }
  }

  void apply(int dim, const std::vector<__int64_t> &shape,
             const std::vector<__int64_t> &strides, __int64_t &offset,
             std::vector<__int64_t> &new_shape,
             std::vector<__int64_t> &new_strides) const override {
    const __int64_t dim_size = shape[dim];
    __int64_t start = start_;
    __int64_t end = end_;
    __int64_t step = step_;

    __int64_t final_start = 0;
    __int64_t final_len = 0;

    if (step > 0) {
      if (start < 0)
        start += dim_size;
      start = std::clamp(start, static_cast<__int64_t>(0), dim_size);

      if (end < 0)
        end += dim_size;
      end = std::clamp(end, static_cast<__int64_t>(0), dim_size);

      if (end > start) {
        final_len = (end - start + step - 1) / step;
      } else {
        final_len = 0;
      }
      final_start = start;
    } else {
      if (start < 0)
        start += dim_size;
      start = std::clamp(start, static_cast<__int64_t>(-1), dim_size - 1);

      if (end < 0)
        end += dim_size;
      end = std::clamp(end, static_cast<__int64_t>(-1), dim_size - 1);

      if (end < start) {
        final_len = (start - end + (-step) - 1) / (-step);
      } else {
        final_len = 0;
      }
      final_start = start;
    }

    new_shape.push_back(final_len);
    new_strides.push_back(strides[dim] * step);
    if (final_len > 0) {
      offset += strides[dim] * final_start;
    }
  }

private:
  __int64_t start_, end_, step_;
};

class EllipsisIndex : public IndexStrategy {
public:
  void apply(int dim, const std::vector<__int64_t> &shape,
             const std::vector<__int64_t> &strides, __int64_t &offset,
             std::vector<__int64_t> &new_shape,
             std::vector<__int64_t> &new_strides) const override {
    new_shape.push_back(shape[dim]);
    new_strides.push_back(strides[dim]);
  }
};

class NewAxisIndex : public IndexStrategy {
public:
  void apply(int dim, const std::vector<__int64_t> &shape,
             const std::vector<__int64_t> &strides, __int64_t &offset,
             std::vector<__int64_t> &new_shape,
             std::vector<__int64_t> &new_strides) const override {
    new_shape.push_back(1);
    new_strides.push_back(0);
  }
};

#endif

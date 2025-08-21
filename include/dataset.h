#pragma once

#ifndef NAWAH_DATASET_H
#define NAWAH_DATASET_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "indexing.h"
#include "tensor.h"

class TensorDataset {
private:
  std::shared_ptr<Tensor> data_tensor_;

public:
  TensorDataset(const std::vector<Tensor> &input_tensors) {
    if (input_tensors.empty()) {
      data_tensor_ = nullptr;
      return;
    }

    if (input_tensors.size() == 1) {
      data_tensor_ = std::make_shared<Tensor>(input_tensors[0].unsqueeze(0));
    } else {
      std::vector<Tensor> tensors_to_stack;
      tensors_to_stack.reserve(input_tensors.size());
      for (const auto &t : input_tensors) {
        tensors_to_stack.push_back(t.unsqueeze(0));
      }

      data_tensor_ = std::make_shared<Tensor>(Tensor::cat(tensors_to_stack, 0));
    }
  }

  explicit TensorDataset(std::shared_ptr<Tensor> data)
      : data_tensor_(std::move(data)) {}

  explicit TensorDataset(Tensor data)
      : data_tensor_(std::make_shared<Tensor>(std::move(data))) {}

  size_t __len__() const {
    if (!data_tensor_) {
      return 0;
    }
    if (data_tensor_->ndim() == 0) {
      return 1;
    }
    return static_cast<size_t>(data_tensor_->shape()[0]);
  }

  Tensor __getitem__(size_t idx) {
    if (!data_tensor_) {
      throw std::runtime_error("TensorDataset is empty. Cannot get item.");
    }
    if (idx >= __len__()) {
      throw std::out_of_range("Index " + std::to_string(idx) +
                              " out of bounds for TensorDataset of size " +
                              std::to_string(__len__()) + ".");
    }

    std::vector<std::shared_ptr<IndexStrategy>> strategies;
    strategies.reserve(data_tensor_->ndim());

    strategies.push_back(
        std::make_shared<IntegerIndex>(static_cast<__int64_t>(idx)));

    for (size_t i = 1; i < data_tensor_->ndim(); ++i) {
      strategies.push_back(std::make_shared<FullSlice>());
    }

    return data_tensor_->get_item(strategies);
  }
};

#endif

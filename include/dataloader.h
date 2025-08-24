#ifndef NAWAH_DATALOADER_H
#define NAWAH_DATALOADER_H

#include "tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <thread>
#include <vector>

namespace py = pybind11;

class DataLoader {
  py::object dataset_;
  bool shuffle_;
  size_t dataset_size_;
  size_t current_idx_ = 0;
  int batch_size_;
  int num_workers_;
  std::vector<size_t> indices_;

  void shuffle_indices() {
    indices_.resize(dataset_size_);

    for (int i = 0; i < dataset_size_; ++i) {
      indices_[i] = i;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(indices_.begin(), indices_.end(), gen);
  }

public:
  DataLoader(py::object dataset, int batch_size, int num_workers, bool shuffle)
      : dataset_(dataset), batch_size_(batch_size), num_workers_(num_workers),
        shuffle_(shuffle) {
    dataset_size_ = py::len(dataset_);

    if (shuffle_) {
      shuffle_indices();
    }
  }

  size_t size() const { return dataset_size_; }

  py::tuple next_batch() {
    if (current_idx_ >= dataset_size_) {
      current_idx_ = 0;
      if (shuffle_) {
        shuffle_indices();
      }
      throw py::stop_iteration();
    }

    size_t end_idx = std::min(current_idx_ + batch_size_, dataset_size_);
    std::vector<Tensor> batch;

    if (!shuffle_) {
      py::int_ start_idx(current_idx_);
      py::int_ end_idx_py(end_idx);
      py::object none_obj = py::none();
      py::slice slice(start_idx, end_idx_py, none_obj);

      batch =
          py::cast<std::vector<Tensor>>(dataset_.attr("__getitem__")(slice));
    } else {
      std::vector<std::vector<Tensor>> individual_batch_items;
      for (size_t i = current_idx_; i < end_idx; ++i) {
        py::int_ idx(indices_[i]);
        std::vector<Tensor> curr =
            py::cast<std::vector<Tensor>>(dataset_.attr("__getitem__")(idx));
        individual_batch_items.push_back(curr);
      }

      if (individual_batch_items.empty()) {
          py::tuple empty_result(0);
          return empty_result;
      }

      size_t num_components_per_item = individual_batch_items[0].size();
      std::vector<std::vector<Tensor>> stacked_tensors(num_components_per_item);

      for (size_t i = 0; i < individual_batch_items.size(); ++i) {
        for (size_t j = 0; j < num_components_per_item; ++j) {
          stacked_tensors[j].push_back(individual_batch_items[i][j]);
        }
      }

      for (int i = 0; i < stacked_tensors.size(); ++i) {
        batch.push_back(Tensor::stack(stacked_tensors[i], 0));
      }
    }
    current_idx_ = end_idx;

    py::tuple result(batch.size());

    for (int i = 0; i < batch.size(); ++i) {
      result[i] = batch[i];
    }
    return result;
  }

  DataLoader &iter() { return *this; }

  ~DataLoader() {}
};

#endif

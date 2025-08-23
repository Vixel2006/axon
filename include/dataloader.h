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

  Tensor next_batch() {
    if (current_idx_ >= dataset_size_) {
      current_idx_ = 0;
      if (shuffle_) {
        shuffle_indices();
      }
      throw py::stop_iteration();
    }

    size_t end_idx = std::min(current_idx_ + batch_size_, dataset_size_);

    py::int_ start_idx(current_idx_);
    py::int_ end_idx_py(end_idx);
    py::object none_obj = py::none();
    py::slice slice(start_idx, end_idx_py, none_obj);

    Tensor batch = py::cast<Tensor>(dataset_.attr("__getitem__")(slice));
    current_idx_ += end_idx;
    return batch;
  }

  DataLoader &iter() { return *this; }

  ~DataLoader() {}
};

#endif

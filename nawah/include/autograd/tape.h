#ifndef NAWAH_AUTOGRAD_TAPE_H
#define NAWAH_AUTOGRAD_TAPE_H

#include <vector>
#include <functional>

class Tensor;

struct Tape {
  std::vector<Tensor> prev;
  std::function<void(const Tensor&, std::vector<Tensor>&)> backward_fn;

  Tape() = default;
  ~Tape() = default;
};

#endif

#ifndef NAWAH_AUTOGRAD_TAPE_H
#define NAWAH_AUTOGRAD_TAPE_H

#include <vector>

class Tensor;

struct Tape {
  std::vector<Tensor> prev;
  char op;

  Tape() = default;
  ~Tape() = default;
};

#endif

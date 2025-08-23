#ifndef NAWAH_OPTIMIZERS_H
#define NAWAH_OPTIMIZERS_H
#include <memory>
#include <vector>

class Tensor;

struct Optimizers {
  virtual void SGD(std::vector<std::shared_ptr<Tensor>> &params, float lr) = 0;
  virtual void zero_grad(std::vector<std::shared_ptr<Tensor>> &params) = 0;
};

struct CpuOptimizers : Optimizers {
  void SGD(std::vector<std::shared_ptr<Tensor>> &params, float lr) override;
  void zero_grad(std::vector<std::shared_ptr<Tensor>> &params) override;
};

struct CudaOptimizers : Optimizers {
  void SGD(std::vector<std::shared_ptr<Tensor>> &params, float lr) override;
  void zero_grad(std::vector<std::shared_ptr<Tensor>> &params) override;
};

#endif

#ifndef NAWAH_OPTIMIZERS_H
#define NAWAH_OPTIMIZERS_H

class Tensor;

struct Optimizers {
  virtual void SGD(std::vector<std::shared_ptr<Tensor>>& params, float lr) = 0;
};

struct CpuOptimizers: Optimizers {
  void SGD(std::vector<std::shared_ptr<Tensor>>& params, float lr) override;
};

struct CudaOptimizers: Optimizers {
  void SGD(std::vector<std::shared_ptr<Tensor>>& params, float lr) override;
};

#endif

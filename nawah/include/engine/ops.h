#ifndef NAWAH_OPS_H
#define NAWAH_OPS_H

class Tensor;

struct Ops {
  virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor sub(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor mul(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor sum(const Tensor& a, int dim, bool keepdim) = 0;
  virtual Tensor mean(const Tensor& a, int dim, bool keepdim) = 0;
  virtual Tensor relu(const Tensor& t, float leakage) = 0;
};

struct CpuOps: Ops {
  Tensor add(const Tensor& a, const Tensor& b) override;
  Tensor sub(const Tensor& a, const Tensor& b) override;
  Tensor mul(const Tensor& a, const Tensor& b) override;
  Tensor matmul(const Tensor& a, const Tensor& b) override;
  Tensor sum(const Tensor& a, int dim, bool keepdim) override;
  Tensor mean(const Tensor& a, int dim, bool keepdim) override;
  Tensor relu(const Tensor& t, float leakage) override;
};

struct CudaOps: Ops {
  Tensor add(const Tensor& a, const Tensor& b) override;
  Tensor sub(const Tensor& a, const Tensor& b) override;
  Tensor mul(const Tensor& a, const Tensor& b) override;
  Tensor matmul(const Tensor& a, const Tensor& b) override;
  Tensor sum(const Tensor& a, int dim, bool keepdim) override;
  Tensor mean(const Tensor& a, int dim, bool keepdim) override;
  Tensor relu(const Tensor& t, float leakage) override;
};

#endif

#ifndef NAWAH_OPS_H
#define NAWAH_OPS_H
#include "autograd/ops.h"

class Tensor;

struct Ops {
  virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor add(const Tensor& a, float scalar) = 0;
  virtual Tensor sub(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor sub(const Tensor& a, float scalar) = 0;
  virtual Tensor mul(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor mul(const Tensor& a, float scalar) = 0;
  virtual Tensor div(const Tensor &numerator, float denominator) = 0;
  virtual Tensor div(const Tensor &numerator, const Tensor& denominator) = 0;
  virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;
  virtual Tensor sum(const Tensor& a, int dim, bool keepdim) = 0;
  virtual Tensor mean(const Tensor& a, int dim, bool keepdim) = 0;
  virtual Tensor relu(const Tensor& t) = 0;
  virtual Tensor log(const Tensor& t) = 0;
  virtual Tensor exp(const Tensor& t) = 0;
  virtual Tensor pow(const Tensor& base, float exponent) = 0;
  virtual Tensor pow(const Tensor& base, const Tensor& exponent) = 0;
  virtual Tensor softmax(const Tensor &a) = 0;
  virtual Tensor neg(const Tensor& a) = 0;
  virtual Tensor conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) = 0;
  virtual void fill_zeros(Tensor& t) const = 0;
  virtual void fill_ones(Tensor& t) const = 0;
  virtual void fill_uniform(Tensor& t) const = 0;
  virtual void fill_randn(Tensor& t) const = 0;
};

struct CpuOps: Ops {
  Tensor add(const Tensor& a, const Tensor& b) override;
  Tensor add(const Tensor& a, float scalar) override;
  Tensor sub(const Tensor& a, const Tensor& b) override;
  Tensor sub(const Tensor& a, float scalar) override;
  Tensor mul(const Tensor& a, const Tensor& b) override;
  Tensor mul(const Tensor& a, float scalar) override;
  Tensor div(const Tensor &numerator, float denominator) override;
  Tensor div(const Tensor &numerator, const Tensor& denominator) override;
  Tensor matmul(const Tensor& a, const Tensor& b) override;
  Tensor sum(const Tensor& a, int dim, bool keepdim) override;
  Tensor mean(const Tensor& a, int dim, bool keepdim) override;
  Tensor relu(const Tensor& t) override;
  Tensor log(const Tensor& t) override;
  Tensor exp(const Tensor& t) override;
  Tensor pow(const Tensor& base, float exponent) override;
  Tensor pow(const Tensor& base, const Tensor& exponent) override;
  Tensor softmax(const Tensor &a) override;
  Tensor neg(const Tensor &a) override;
  Tensor conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) override;
  void fill_ones(Tensor& t) const override;
  void fill_zeros(Tensor& t) const override;
  void fill_uniform(Tensor& t) const override;
  void fill_randn(Tensor& t) const override;
};

struct CudaOps: Ops {
  Tensor add(const Tensor& a, const Tensor& b) override;
  Tensor add(const Tensor& a, float scalar) override;
  Tensor sub(const Tensor& a, const Tensor& b) override;
  Tensor sub(const Tensor& a, float scalar) override;
  Tensor mul(const Tensor& a, const Tensor& b) override;
  Tensor mul(const Tensor& a, float scalar) override;
  Tensor div(const Tensor &numerator, float denominator) override;
  Tensor div(const Tensor &numerator, const Tensor& denominator) override;
  Tensor matmul(const Tensor& a, const Tensor& b) override;
  Tensor sum(const Tensor& a, int dim, bool keepdim) override;
  Tensor mean(const Tensor& a, int dim, bool keepdim) override;
  Tensor relu(const Tensor& t) override;
  Tensor log(const Tensor& t) override;
  Tensor exp(const Tensor& t) override;
  Tensor pow(const Tensor& base, float exponent) override;
  Tensor pow(const Tensor& base, const Tensor& exponent) override;
  Tensor softmax(const Tensor &a) override;
  Tensor neg(const Tensor& a) override;
  Tensor conv2d(const Tensor& a, const Tensor& kernel, int stride, int padding) override;
  void fill_ones(Tensor& t) const override;
  void fill_zeros(Tensor& t) const override;
  void fill_uniform(Tensor& t) const override;
  void fill_randn(Tensor& t) const override;
};

#endif

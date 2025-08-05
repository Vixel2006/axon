#ifndef NAWAH_AUTOGRAD_OPS_H
#define NAWAH_AUTOGRAD_OPS_H
#include <vector>

class Tensor;

struct CpuAutograd {
  static void add(const Tensor& out, std::vector<Tensor>& prev);
  static void sub(const Tensor& out, std::vector<Tensor>& prev);
  static void mul(const Tensor& out, std::vector<Tensor>& prev);
  static void matmul(const Tensor& out, std::vector<Tensor>& prev);
  static void div(const Tensor& out, std::vector<Tensor>& prev);
  static void log(const Tensor& out, std::vector<Tensor>& prev);
  static void exp(const Tensor& out, std::vector<Tensor>& prev);
  static void pow(const Tensor& out, std::vector<Tensor>& prev);
};

struct CudaAutograd {
  static void add(const Tensor& out, std::vector<Tensor>& prev);
  static void sub(const Tensor& out, std::vector<Tensor>& prev);
  static void mul(const Tensor& out, std::vector<Tensor>& prev);
  static void matmul(const Tensor& out, std::vector<Tensor>& prev);
  static void div(const Tensor& out, std::vector<Tensor>& prev);
  static void log(const Tensor& out, std::vector<Tensor>& prev);
  static void exp(const Tensor& out, std::vector<Tensor>& prev);
  static void pow(const Tensor& out, std::vector<Tensor>& prev);
};


#endif

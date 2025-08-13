#ifndef NAWAH_AUTOGRAD_OPS_H
#define NAWAH_AUTOGRAD_OPS_H
#include <vector>

class Tensor;

struct CpuAutograd {
  static void add(Tensor& out, std::vector<Tensor>& prev);
  static void sub(Tensor& out, std::vector<Tensor>& prev);
  static void mul(Tensor& out, std::vector<Tensor>& prev);
  static void matmul(Tensor& out, std::vector<Tensor>& prev);
  static void div(Tensor& out, std::vector<Tensor>& prev);
  static void log(Tensor& out, std::vector<Tensor>& prev);
  static void exp(Tensor& out, std::vector<Tensor>& prev);
  static void pow(Tensor& out, std::vector<Tensor>& prev);
  static void sum(Tensor& out, std::vector<Tensor>& prev);
  static void mean(Tensor& out, std::vector<Tensor>& prev);
  static void relu(Tensor& out, std::vector<Tensor>& prev);
  static void conv2d(Tensor& out, std::vector<Tensor>& prev, int stride, int padding);
  static void flatten(Tensor& out, std::vector<Tensor>& prev);
};

struct CudaAutograd {
  static void add(Tensor& out, std::vector<Tensor>& prev);
  static void sub(Tensor& out, std::vector<Tensor>& prev);
  static void mul(Tensor& out, std::vector<Tensor>& prev);
  static void matmul(Tensor& out, std::vector<Tensor>& prev);
  static void div(Tensor& out, std::vector<Tensor>& prev);
  static void log(Tensor& out, std::vector<Tensor>& prev);
  static void exp(Tensor& out, std::vector<Tensor>& prev);
  static void pow(Tensor& out, std::vector<Tensor>& prev);
  static void sum(Tensor& out, std::vector<Tensor>& prev);
  static void mean(Tensor& out, std::vector<Tensor>& prev);
  static void relu(Tensor& out, std::vector<Tensor>& prev);
  static void conv2d(Tensor& out, std::vector<Tensor>& prev, int stride, int padding);
  static void flatten(Tensor& out, std::vector<Tensor>& prev);
};


#endif

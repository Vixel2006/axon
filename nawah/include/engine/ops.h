#ifndef NAWAH_OPS_H
#define NAWAH_OPS_H

class Tensor;

struct AddImpl {
    static Tensor forward_cpu(const Tensor& a, const Tensor& b);
    static Tensor forward_gpu(const Tensor& a, const Tensor& b);
    static Tensor backward_cpu(const Tensor& a, const Tensor& b);
    static Tensor backward_gpu(const Tensor& a, const Tensor& b);
};

struct SubImpl {
    static Tensor forward_cpu(const Tensor& a, const Tensor& b);
    static Tensor forward_gpu(const Tensor& a, const Tensor& b);
    static Tensor backward_cpu(const Tensor& a, const Tensor& b);
    static Tensor backward_gpu(const Tensor& a, const Tensor& b);
};

struct MulImpl {
    static Tensor forward_cpu(const Tensor& a, const Tensor& b);
    static Tensor forward_gpu(const Tensor& a, const Tensor& b);
    static Tensor backward_cpu(const Tensor& a, const Tensor& b);
    static Tensor backward_gpu(const Tensor& a, const Tensor& b);
};

struct MatmulImpl {
    static Tensor forward_cpu(const Tensor& a, const Tensor& b);
    static Tensor forward_gpu(const Tensor& a, const Tensor& b);
    static Tensor backward_cpu(const Tensor& a, const Tensor& b);
    static Tensor backward_gpu(const Tensor& a, const Tensor& b);
};

struct SumImpl {
    static Tensor forward_cpu(const Tensor& a, int dim, bool keepdim);
    static Tensor forward_gpu(const Tensor& a, int dim, bool keepdim);
    static Tensor backward_cpu(const Tensor& a);
    static Tensor backward_gpu(const Tensor& a);
};

struct MeanImpl {
    static Tensor forward_cpu(const Tensor& a, int dim, bool keepdim);
    static Tensor forward_gpu(const Tensor& a, int dim, bool keepdim);
    static Tensor backward_cpu(const Tensor& a);
    static Tensor backward_gpu(const Tensor& a);
};

#endif

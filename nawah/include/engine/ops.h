#ifndef NAWAH_OPS_H
#define NAWAH_OPS_H

class Tensor;

struct AddImpl {
    static Tensor cpu(const Tensor& a, const Tensor& b);
    static Tensor gpu(const Tensor& a, const Tensor& b);
};

struct SubImpl {
    static Tensor cpu(const Tensor& a, const Tensor& b);
    static Tensor gpu(const Tensor& a, const Tensor& b);
};

struct MulImpl {
    static Tensor cpu(const Tensor& a, const Tensor& b);
    static Tensor gpu(const Tensor& a, const Tensor& b);
};

struct MatmulImpl {
    static Tensor cpu(const Tensor& a, const Tensor& b);
    static Tensor gpu(const Tensor& a, const Tensor& b);
};

struct SumImpl {
    static Tensor cpu(const Tensor& a, int dim, bool keepdim);
    static Tensor gpu(const Tensor& a, int dim, bool keepdim);
};

struct MeanImpl {
    static Tensor cpu(const Tensor& a, int dim, bool keepdim);
    static Tensor gpu(const Tensor& a, int dim, bool keepdim);
};

#endif

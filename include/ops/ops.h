#ifndef NAWAH_OPS_H
#define NAWAH_OPS_H

template<typename T>
class Tensor;

#include "ops/add.h"
#include "ops/sub.h"
#include "ops/mul.h"
#include "ops/matmul.h"

class Union {
    public:
    bool is_scalar;
    float scalar;
    Tensor tensor;

    Union(float scalar) : is_scalar(true), scalar(scalar) {}
    Union(Tensor tensor) : is_scalar(false), tensor(tensor) {}
};

struct OpTrait {
    virtual Tensor apply(const Union& a, const Union& b) = 0;
};

struct AddTrait : public OpTrait {
    Tensor apply(const Union& a, const Union& b) override;
};

struct SubTrait : public OpTrait {
    Tensor apply(const Union& a, const Union& b) override;
};

struct MulTrait : public OpTrait {
    Tensor apply(const Union& a, const Union& b) override;
};

struct MatMulTrait : public OpTrait {
    Tensor apply(const Union& a, const Union& b) override;
};


#endif
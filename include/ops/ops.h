#ifndef NAWAH_OPS_H
#define NAWAH_OPS_H

#include "ops/add.h"
#include "ops/sub.h"
#include "ops/mul.h"
#include "ops/matmul.h"
#include "tensor.h"

struct OpTrait {
    virtual Tensor apply(const Tensor& a, const Tensor& b) = 0;
};

struct AddTrait : public OpTrait {
    Tensor apply(const Tensor& a, const Tensor& b) override;
};

struct SubTrait : public OpTrait {
    Tensor apply(const Tensor& a, const Tensor& b) override;
};

/*
struct MulTrait : public OpTrait {
    Tensor apply(const Tensor& a, const Tensor& b) override;
};
*/

struct MatMulTrait : public OpTrait {
    Tensor apply(const Tensor& a, const Tensor& b) override;
};


#endif
#ifndef IDRAK_OPS_H
#define IDRAK_OPS_H

#include "tensor.h"

typedef void (*UnaryOpFn)(Tensor *in, Tensor *out);
typedef void (*BinaryOpFn)(Tensor *a, Tensor *b, Tensor *out);
void add_op(Tensor *a, Tensor *b, Tensor *out);
void sub_op(Tensor *a, Tensor *b, Tensor *out);
void mul_op(Tensor *a, Tensor *b, Tensor *out);
void div_op(Tensor *a, Tensor *b, Tensor *out);
void pow_op(Tensor *a, Tensor *b, Tensor *out);
typedef void (*BinaryOpScalarFn)(Tensor *a, float b, Tensor *out);
void add_scalar_op(Tensor *a, float b, Tensor *out);
void sub_scalar_op(Tensor *a, float b, Tensor *out);
void mul_scalar_op(Tensor *a, float b, Tensor *out);
void div_scalar_op(Tensor *a, float b, Tensor *out);
void pow_scalar_op(Tensor *a, float b, Tensor *out);
typedef void (*ReductionOpFn)(Tensor *in, Tensor *out, int axis);
typedef void (*MovementOpFn)(Tensor *in, Tensor *out, void *args);

typedef struct {
  UnaryOpFn relu;
  UnaryOpFn log;
  UnaryOpFn exp;
  UnaryOpFn softmax;
  UnaryOpFn abs;
  UnaryOpFn neg;
  UnaryOpFn tanh;
  UnaryOpFn sigmoid;
} UniaryOps;

typedef struct {
  BinaryOpFn add;
  BinaryOpFn sub;
  BinaryOpFn mul;
  BinaryOpFn div;
  BinaryOpFn pow;
  BinaryOpFn matmul;
} BinaryOps;

typedef struct {
  BinaryOpScalarFn add;
  BinaryOpScalarFn sub;
  BinaryOpScalarFn mul;
  BinaryOpScalarFn div;
  BinaryOpScalarFn pow;
} BinaryScalarOps;

typedef struct {
  ReductionOpFn sum;
  ReductionOpFn mean;
  ReductionOpFn max;
} ReductionOps;

typedef struct {
  MovementOpFn view;
  MovementOpFn squeeze;
  MovementOpFn unsqueeze;
  MovementOpFn transpose;
  MovementOpFn expand;
  MovementOpFn broadcast;
  MovementOpFn flatten;
} MovementOps;

#endif

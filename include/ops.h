#ifndef NAWAH_OPS_H
#define NAWAH_OPS_H

#include "tensor.h"

typedef void (*UnaryOpFn)(Tensor *in, Tensor *out);
typedef void (*BinaryOpFn)(Tensor *a, Tensor *b, Tensor *out);
typedef void (*BinaryOpScalarFn)(Tensor *a, float b, Tensor *out);
typedef void (*ReductionOpFn)(Tensor *in, Tensor *out);
typedef void (*MovementOpFn)(Tensor *in, Tensor *out, void *args);

typedef struct {
  UnaryOpFn relu;
  UnaryOpFn log;
  UnaryOpFn exp;
  UnaryOpFn softmax;
  UnaryOpFn abs;
  UnaryOpFn neg;
} UniaryOps;

typedef struct {
  BinaryOpFn add;
  BinaryOpFn sub;
  BinaryOpFn mul;
  BinaryOpFn div;
  BinaryOpFn matmul;
} BinaryOps;

typedef struct {
  BinaryOpScalarFn add;
  BinaryOpScalarFn sub;
  BinaryOpScalarFn mul;
  BinaryOpScalarFn div;
} BinaryScalarOps;

typedef struct {
  ReductionOpFn sum;
  ReductionOpFn mean;
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

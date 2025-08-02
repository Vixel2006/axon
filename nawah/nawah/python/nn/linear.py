from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__(name="Linear")

        self.add_parameter("weight", Tensor.randn(in_features, out_features))
        self.add_parameter("bias", Tensor.randn(out_features))
    
    def forward(self, x):
        w = self._parameter["weight"]
        b = self._parameter["bias"]

        return x @ w + b


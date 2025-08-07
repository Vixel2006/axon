import cnawah as nw

def tanh():
    def tanh_fn_(x):
        if x is None:
            raise ValueError("Input to tanh cannot be None.")

        if not hasattr(x, "shape"):
            raise TypeError(f"Input to tanh must have a 'shape' attribute, got {type(x)}.")

        if not hasattr(x, "requires_grad"):
            raise TypeError("Input to tanh must be a Nawah tensor with 'requires_grad' attribute.")

        try:
            numerator = nw.exp(x) - nw.exp(-x)
            denominator = nw.exp(x) + nw.exp(-x)
            return numerator / denominator
        except Exception as e:
            raise RuntimeError(f"Failed to compute tanh due to internal error: {str(e)}")

    return {
        "name": "Tanh",
        "fn": tanh_fn_
    }

def relu():
    def relu_fn_(x):
        if x is None:
            raise ValueError("Input to tanh cannot be None.")

        if not hasattr(x, "shape"):
            raise TypeError(f"Input to tanh must have a 'shape' attribute, got {type(x)}.")

        if not hasattr(x, "requires_grad"):
            raise TypeError("Input to tanh must be a Nawah tensor with 'requires_grad' attribute.")

        try:
            return nw.relu(x)
        except Exception as e:
            raise RuntimeError(f"Failed to compute tanh due to internal error: {str(e)}")

    return {
        "name": "ReLU",
        "fn": relu_fn_
    }


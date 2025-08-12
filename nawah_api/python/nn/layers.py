import cnawah as nw

def linear(in_dims: int, out_dims: int, has_bias: bool = True):
    if not isinstance(in_dims, int) or in_dims <= 0:
        raise ValueError(f"[linear] 'in_dims' must be a positive integer, got {in_dims}")
    if not isinstance(out_dims, int) or out_dims <= 0:
        raise ValueError(f"[linear] 'out_dims' must be a positive integer, got {out_dims}")
    if not isinstance(has_bias, bool):
        raise TypeError(f"[linear] 'has_bias' must be a boolean, got {type(has_bias).__name__}")

    params = {}


    def linear_fn_(x):
        nonlocal params
        if not hasattr(x, "shape"):
            raise TypeError(f"[linear_fn_] input must have a 'shape' attribute (likely a Tensor), got {type(x).__name__}")
        if len(x.shape) != 2:
            raise ValueError(f"[linear_fn_] expected input shape (batch_size, in_dims), got shape {x.shape}")
        if x.shape[1] != in_dims:
            raise ValueError(f"[linear_fn_] input dim mismatch: expected {in_dims}, got {x.shape[1]}")
        
        batch_size = x.shape[0]

        if "w" not in params:
            params = {
                "w": nw.randn([batch_size, in_dims, out_dims], requires_grad=True)
            }

        out = x @ params['w']
        
        if has_bias:
            if "b" not in params:
                params['b'] = nw.zeros([batch_size, out_dims], requires_grad=True)

            out = out + params['b']

        return out

    return {
        "name": "linear",
        "in_dims": in_dims,
        "out_dims": out_dims,
        "params": params,
        "fn": linear_fn_
    }

def conv2d(in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int = 1, padding: int = 0, has_bias: bool = False):
    assert isinstance(in_channels, int) and in_channels > 0, "in_channels must be a positive integer."
    assert isinstance(out_channels, int) and out_channels > 0, "out_channels must be a positive integer."
    assert isinstance(kernel_size, tuple) and len(kernel_size) == 2 and kernel_size[0] > 0 and kernel_size[1] > 0, "kernel_size must be a tuple of two positive integers."
    assert isinstance(stride, int) and stride > 0, "stride must be a positive integer."
    assert isinstance(padding, int) and padding >= 0, "padding must be a non-negative integer."
    assert isinstance(has_bias, bool), "has_bias must be a boolean."

    params = {
        "W": nw.uniform([out_channels, in_channels, *kernel_size], requires_grad=True),
    }

    if has_bias:
        params["b"] = nw.zeros([1, out_channels, 1, 1], requires_grad=True)

    def conv2d_fn_(x):
        if not hasattr(x, "shape"):
            raise TypeError(f"[conv2d_fn_] input must have a 'shape' attribute, got {type(x).__name__}")
        if len(x.shape) != 4:
            raise ValueError(f"[conv2d_fn_] expected a 4D input tensor (N, C, H, W), got shape {x.shape}")
        if x.shape[1] != in_channels:
            raise ValueError(f"[conv2d_fn_] input channel mismatch: expected {in_channels}, got {x.shape[1]}")

        out = nw.conv2d(x, params["W"], stride, padding)
        
        if "b" in params:
            out = out + params["b"]
            
        return out

    return {
        "name": "Conv2D",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "params": params,
        "fn": conv2d_fn_
    }

def rnn(input_dim: int, hidden_dim: int, has_bias: bool = True):
    assert isinstance(input_dim, int) and input_dim > 0, "input_dim must be a positive integer."
    assert isinstance(hidden_dim, int) and hidden_dim > 0, "hidden_dim must be a positive integer."
    assert isinstance(has_bias, bool), "has_bias must be a boolean."

    params = {
        "W_ih": nw.uniform([hidden_dim, input_dim], requires_grad=True),
        "W_hh": nw.uniform([hidden_dim, hidden_dim], requires_grad=True)
    }

    if has_bias:
        params["b"] = nw.zeros([hidden_dim])
    else:
        params["b"] = nw.zeros([hidden_dim])

    def rnn_cell(x, prev=None):
        assert x.ndim == 2, f"Expected input shape (B, input_dim), got {x.shape}"
        B, I = x.shape
        assert I == input_dim, f"Expected input dim {input_dim}, got {I}"

        if prev is not None:
            assert prev.ndim == 2, f"Expected hidden state shape (B, hidden_dim), got {prev.shape}"
            assert prev.shape[1] == hidden_dim, f"Expected hidden dim {hidden_dim}, got {prev.shape[1]}"

        x_proj = x @ params["W_ih"].transpose(-1, -2)
        
        if prev is not None:
            h_proj = prev @ params["W_hh"].transpose(-1, -2)
        else:
            h_proj = 0

        out = x_proj + h_proj + params["b"]
        return out >> nw.tanh

    def rnn_fn_(x, h0=None):
        assert x.ndim == 3, f"Expected input shape (B, T, input_dim), got {x.shape}"
        B, T, I = x.shape
        assert I == input_dim, f"Input dimension mismatch. Expected {input_dim}, got {I}"

        if h0 is not None:
            assert h0.shape == (B, hidden_dim), f"Initial hidden state shape must be (B, {hidden_dim}), got {h0.shape}"
        else:
            h0 = nw.zeros([B, hidden_dim])

        h_t = h0
        outputs = []

        for t in range(T):
            x_t = x[:, t, :]
            h_t = rnn_cell(x_t, h_t)
            outputs.append(h_t)
        
        return nw.stack(outputs, axis=1)

    return {
        "name": "RNN",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "params": params,
        "fn": rnn_fn_
    }

def layer_norm(normalized_shape, eps=1e-5):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    params = {
        "gamma": nw.ones(normalized_shape, requires_grad=True), # Gain
        "beta": nw.zeros(normalized_shape, requires_grad=True)  # Bias
    }

    def layer_norm_fn_(x):
        dims_to_normalize = tuple(range(x.ndim - len(normalized_shape), x.ndim))

        mean = nw.mean(x, axis=dims_to_normalize, keepdims=True)
        var = nw.mean((x - mean) ** 2, axis=dims_to_normalize, keepdims=True)

        x_normalized = (x - mean) * ((var + eps) ** -0.5)

        return params["gamma"] * x_normalized + params["beta"]

    return {
        "name": "LayerNorm",
        "params": params,
        "fn": layer_norm_fn_
    }

def flatten():
    def flatten_fn_(x):
        return x.view([x.shape[0], -1])

    return {
        "name": "Flatten",
        "params": {},
        "fn": flatten_fn_
    }

def scaled_dot_product_attention():
    def attention_fn_(q, k, v):
        d_k = k.shape[-1]

        scores = nw.matmul(q, k.transpose(-1, -2)) * (d_k ** -0.5)

        attention_weights = softmax(axis=-1)["fn"](scores)

        output = nw.matmul(attention_weights, v)
        return output

    return {
        "name": "ScaledDotProductAttention",
        "params": {},
        "fn": attention_fn_
    }


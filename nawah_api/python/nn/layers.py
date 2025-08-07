import nawah_api as nw

def linear(in_dims: int, out_dims: int, has_bias: bool = True):
    if not isinstance(in_dims, int) or in_dims <= 0:
        raise ValueError(f"[linear] 'in_dims' must be a positive integer, got {in_dims}")
    if not isinstance(out_dims, int) or out_dims <= 0:
        raise ValueError(f"[linear] 'out_dims' must be a positive integer, got {out_dims}")
    if not isinstance(has_bias, bool):
        raise TypeError(f"[linear] 'has_bias' must be a boolean, got {type(has_bias).__name__}")

    params = {
        "w": nw.randn([in_dims, out_dims], requires_grad=True)
    }

    if has_bias:
        params['b'] = nw.ones([1, out_dims], requires_grad=True)

    def linear_fn_(x):
        if not hasattr(x, "shape"):
            raise TypeError(f"[linear_fn_] input must have a 'shape' attribute (likely a Tensor), got {type(x).__name__}")
        if len(x.shape) != 2:
            raise ValueError(f"[linear_fn_] expected input shape (batch_size, in_dims), got shape {x.shape}")
        if x.shape[1] != in_dims:
            raise ValueError(f"[linear_fn_] input dim mismatch: expected {in_dims}, got {x.shape[1]}")
        
        out = x @ params['w']
        if 'b' in params:
            out = out + params['b']

        return out

    return {
        "name": "linear",
        "in_dims": in_dims,
        "out_dims": out_dims,
        "params": params,
        "fn": linear_fn_
    }

def rnn(input_dim: int, hidden_dim: int, has_bias: bool = True):
    assert isinstance(input_dim, int) and input_dim > 0, "input_dim must be a positive integer."
    assert isinstance(hidden_dim, int) and hidden_dim > 0, "hidden_dim must be a positive integer."
    assert isinstance(has_bias, bool), "has_bias must be a boolean."

    params = {
        "W_ih": nw.uniform([hidden_dim, input_dim]),
        "W_hh": nw.uniform([hidden_dim, hidden_dim])
    }

    if has_bias:
        params["b"] = nw.ones([hidden_dim])
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
            h_proj = prev @ params["W_hh"]
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


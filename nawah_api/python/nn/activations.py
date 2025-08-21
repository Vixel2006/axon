import cnawah as nw


def tanh():
    def tanh_fn_(x):
        e_pos = nw.exp(x)
        e_neg = nw.exp(-x)
        return (e_pos - e_neg) * ((e_pos + e_neg) ** -1)

    return {"name": "Tanh", "params": {}, "fn": tanh_fn_}


def sigmoid():
    def sigmoid_fn_(x):
        return (nw.ones_like(x) + nw.exp(-x)) ** -1

    return {"name": "Sigmoid", "params": {}, "fn": sigmoid_fn_}


def softmax(axis=-1):
    def softmax_fn_(x):
        exps = nw.exp(x)
        sum_exps = nw.sum(exps, axis=axis, keepdims=True)

        return exps * (sum_exps**-1)

    return {"name": "Softmax", "params": {}, "fn": softmax_fn_}


def relu():
    def relu_fn_(x):
        return nw.relu(x)

    return {"name": "ReLU", "params": {}, "fn": relu_fn_}

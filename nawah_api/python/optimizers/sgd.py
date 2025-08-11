from ..core import Tensor

def sgd(params: list[Tensor], state: dict | None, lr: float) -> (list, dict):
    new_param_data = []
    for param in params:
        if param.grad is not None:
            new_data = param.data - lr * param.grad.data
            new_param_data.append(new_data)
        else:
            new_param_data.append(param.data)

    return new_param_data, {}



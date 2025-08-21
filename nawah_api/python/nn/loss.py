from ..core import Tensor
import cnawah as nw


def mean_squared_error():
    def mse_fn_(y_pred, y_true):
        return nw.mean((y_pred - y_true) ** 2)

    return {"name": "MSELoss", "fn": mse_fn_}


def cross_entropy_loss():
    def cel_fn_(y_pred, y_true, eps=1e-9):
        y_pred = nw.clip(y_pred, eps, 1.0 - eps)
        return -nw.sum(y_true * nw.log(y_pred))

    return {"name": "CrossEntropyLoss", "fn": cel_fn_}

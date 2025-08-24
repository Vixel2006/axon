from ..core import Tensor
import cnawah as nw


def mean_squared_error():
    def mse_fn_(y_pred, y_true):
        out = y_pred - y_true
        return out.mean()

    return {"name": "MSELoss", "fn": mse_fn_}


def cross_entropy_loss():
    def cel_fn_(y_pred, y_true, eps=1e-9):
        y_pred = nw.clip(y_pred, eps, 1.0 - eps)
        return -nw.sum(y_true * nw.log(y_pred))

    return {"name": "CrossEntropyLoss", "fn": cel_fn_}


def bce_with_logits_loss(reduction="mean"):
    def bce_fn_(input: Tensor, target: Tensor) -> Tensor:
        neg_abs_input = -input
        loss = nw.relu(input) - input * target + nw.log(nw.exp(neg_abs_input) + 1.0)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Reduction type '{reduction}' not supported.")

    return {"name": "BCEWithLogitsLoss", "fn": bce_fn_}

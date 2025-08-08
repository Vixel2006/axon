from ..core import Tensor

def reduce(loss_fn):
    def wrapper(pred, target, reduction="none"):
        loss = loss_fn(pred, target)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss
    return wrapper

@reduce
def mse(pred: Tensor, truth: Tensor) -> Tensor:
    loss = (truth - pred) ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

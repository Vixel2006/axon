from idrak.functions import clip, log, mean
from idrak.core.tensor import Tensor

def bce(pred: Tensor, truth: Tensor, reduction: str = "mean", epsilon: float = 1e-7) -> Tensor:
    # Clip predictions to avoid log(0) or log(1) which can lead to NaN/inf
    pred = clip(pred, epsilon, 1.0 - epsilon)
    out = -(truth * log(pred)) - ((1 - truth) * log(1 - pred))
    if reduction == "mean":
        return mean(out)
    return out


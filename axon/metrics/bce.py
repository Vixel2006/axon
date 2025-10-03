from axon.functions import clip, log, mean, from_data
from axon.core.tensor import Tensor

def bce(pred: Tensor, truth: Tensor, reduction: str = "mean", epsilon: float = 1e-6) -> Tensor:
    # Clip predictions to avoid log(0) or log(1) which can lead to NaN/inf
    pred = clip(pred, epsilon, 1.0 - epsilon)
    out = -(truth * log(pred)) - ((1 - truth) * log(1 - pred))

    return mean(out)


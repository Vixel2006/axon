from idrak.functions import clip, log, mean
from idrak.core.tensor import Tensor

def bce(pred: Tensor, truth: Tensor, reduction: str = "mean", epsilon: float = 1e-6) -> Tensor:
    # Clip predictions to avoid log(0) or log(1) which can lead to NaN/inf
    pred = clip(pred, epsilon, 1.0 - epsilon)
    out = -(truth * log(pred)) - ((1 - truth) * log(1 - pred))

    return out

if __name__ == "__main__":
    pred = Tensor((2,2), [[.5, .5], [.5, .5]])
    truth = Tensor((2,2), [[1., 2.], [3., 4.]])

    loss = bce(pred, truth)

    loss.backward()
    print(loss)
    print(loss.grad);print(pred.grad);print(truth.grad)


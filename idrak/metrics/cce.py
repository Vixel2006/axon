from idrak.functions import log_softmax, sum, mean
from idrak.core.tensor import Tensor

def categorical_cross_entropy(logits: Tensor, truth: Tensor) -> Tensor:
    # Compute log_softmax for numerical stability
    log_probs = log_softmax(logits, dim=1)
    # Compute negative log-likelihood
    nll = -sum(truth * log_probs, dim=1)
    return mean(nll)

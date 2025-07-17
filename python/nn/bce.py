# --- file: src/nn/bce.py ---

from ..core import Tensor
from .module import Module


class BCEWithLogitLoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Invalid reduction type: {reduction}. Must be 'mean', 'sum', or 'none'."
            )
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        max_val = logits.relu()

        stable_term = (Tensor(1.0) + (-logits.abs()).exp()).log()

        loss = max_val - logits * target + stable_term

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.sum() / loss.data.size
        else:
            return loss

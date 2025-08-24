from typing import Dict, Any, TYPE_CHECKING
import nawah_api as nw

if TYPE_CHECKING:
    from .runner import Runner


class Callback:
    runner: "Runner" = None

    def on_train_start(self, logs: Dict[str, Any] = None):
        pass

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any] = None):
        pass

    def on_batch_start(self, epoch: int, batch_idx: int, logs: Dict[str, Any] = None):
        pass

    def on_batch_end(self, epoch: int, batch_idx: int, logs: Dict[str, Any] = None):
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        pass

    def on_validation_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        pass

    def on_train_end(self, logs: Dict[str, Any] = None):
        pass

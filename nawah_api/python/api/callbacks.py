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


class SimpleLogger(Callback):
    def on_batch_end(self, epoch: int, batch_idx: int, logs: Dict[str, Any] = None):
        if "loss" in logs and batch_idx % self.runner.config.log_interval == 0:
            total_batches_in_epoch = (
                len(self.runner.train_dataloader)
                if self.runner.train_dataloader
                else "N/A"
            )
            print(
                f"Epoch {epoch+1}/{self.runner.config.epochs} | Batch {batch_idx+1}/{total_batches_in_epoch} | Loss: {logs['loss']:.4f}"
            )

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        log_message = f"Epoch {epoch+1} completed. "
        if "avg_train_loss" in logs:
            log_message += f"Avg Train Loss: {logs['avg_train_loss']:.4f} "
        if "val_loss" in logs:
            log_message += f"Val Loss: {logs['val_loss']:.4f}"
        print(log_message.strip())


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: str = "model_epoch_{epoch:03d}.nawah_model",
        save_best_only: bool = False,
        monitor: str = "val_loss",
    ):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_metric = float("inf") if "loss" in monitor else float("-inf")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current_metric = logs.get(self.monitor)

        should_save = False
        if self.save_best_only:
            if current_metric is not None:
                if "loss" in self.monitor and current_metric < self.best_metric:
                    should_save = True
                elif "accuracy" in self.monitor and current_metric > self.best_metric:
                    should_save = True
        else:
            should_save = True

        if should_save:
            save_path = self.filepath.format(epoch=epoch + 1)
            print(f"Saving model to {save_path}...")
            print(
                "Note: Implement `self.runner.model.save_state_dict(filepath)` in your `nawah_api` `Sequential` (or base Module) for this to work."
            )

            if self.save_best_only and current_metric is not None:
                print(
                    f"(Metric {self.monitor} improved from {self.best_metric:.4f} to {current_metric:.4f})"
                )
                self.best_metric = current_metric

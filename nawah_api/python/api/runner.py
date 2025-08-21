from typing import Callable, List, Dict, Any, Tuple
import nawah_api as nw
from .config import Config
from .callbacks import Callback, SimpleLogger


class Runner:
    def __init__(
        self,
        model,
        train_dataloader,
        optimizer,
        loss_fn,
        config,
        val_dataloader=None,
        callbacks=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = self.config.device

        print(f"Moving model to {self.device}...")
        self.model.to(self.device)

        self.callbacks = callbacks if callbacks is not None else [SimpleLogger()]
        self._initialize_callbacks()

        self.logs: Dict[str, Any] = {}

    def _initialize_callbacks(self):
        for callback in self.callbacks:
            if not isinstance(callback, Callback):
                raise TypeError(
                    f"Callback {type(callback)} is not an instance of callbacks.Callback"
                )
            callback.runner = self

    def _call_callbacks(self, method_name: str, **kwargs):
        for callback in self.callbacks:
            getattr(callback, method_name)(**kwargs, logs=self.logs)

    def _prepare_batch(self, batch: Any) -> Tuple[nw.Tensor, nw.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
        else:
            if isinstance(batch, nw.Tensor):
                inputs = batch.to(self.device)
                targets = inputs
            else:
                raise TypeError(
                    f"Unsupported batch type: {type(batch)}. Batch must be (inputs, targets) tuple of nw.Tensors or a single nw.Tensor."
                )
        return inputs, targets

    def train_step(self, inputs: nw.Tensor, targets: nw.Tensor) -> nw.Tensor:
        self.model.train()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        return loss

    def validation_step(self, inputs: nw.Tensor, targets: nw.Tensor) -> nw.Tensor:
        self.model.eval()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss

    def train(self):
        self._call_callbacks("on_train_start")

        for epoch in range(self.config.epochs):
            self.model.train()
            total_train_loss_scalar = 0.0
            self.logs["current_epoch"] = epoch

            self._call_callbacks("on_epoch_start", epoch=epoch)

            for batch_idx, batch in enumerate(self.train_dataloader):
                self._call_callbacks("on_batch_start", epoch=epoch, batch_idx=batch_idx)

                inputs, targets = self._prepare_batch(batch)

                self.optimizer.zero_grad()
                loss_tensor = self.train_step(inputs, targets)
                self.optimizer.step()

                loss_scalar = loss_tensor.item()
                total_train_loss_scalar += loss_scalar
                self.logs["loss"] = loss_scalar
                self.logs["batch_idx"] = batch_idx

                self._call_callbacks("on_batch_end", epoch=epoch, batch_idx=batch_idx)

            avg_train_loss_scalar = total_train_loss_scalar / len(self.train_dataloader)
            self.logs["avg_train_loss"] = avg_train_loss_scalar

            if self.val_dataloader:
                val_loss_scalar = self.validate()
                self.logs["val_loss"] = val_loss_scalar
                self._call_callbacks("on_validation_epoch_end", epoch=epoch)

            self._call_callbacks("on_epoch_end", epoch=epoch)

        self._call_callbacks("on_train_end")

    def validate(self) -> float:
        self.model.eval()
        total_val_loss_scalar = 0.0

        for batch_idx, batch in enumerate(self.val_dataloader):
            inputs, targets = self._prepare_batch(batch)
            loss_tensor = self.validation_step(inputs, targets)
            total_val_loss_scalar += loss_tensor.item()

        avg_val_loss_scalar = total_val_loss_scalar / len(self.val_dataloader)
        return avg_val_loss_scalar

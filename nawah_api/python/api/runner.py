from typing import Callable, List, Dict, Any, Tuple
import nawah_api as nw
from .config import Config
from .callbacks import Callback


class Runner:
    def __init__(
        self,
        model: nw.Sequential,
        optimizer: nw.Optimizer,
        loss_fn: Callable,
        train_dataloader,
        val_dataloader=None,
        config: Config = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config if config else Config()

    def train(self, training_step: Callable):
        logs = {}

        for epoch in range(self.config.epochs):
            epoch_logs = {"epoch": epoch}

            self.model.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for batch_idx, batch_data in enumerate(self.train_dataloader):
                inputs, targets = batch_data[0], batch_data[1]
                batch_logs = {
                    "batch_idx": batch_idx,
                    "inputs": inputs,
                    "targets": targets,
                }

                self.optimizer.zero_grad()
                loss = training_step(self.model, inputs, targets, self.loss_fn)
                loss.backward()
                self.optimizer.step()

                total_train_loss = loss + total_train_loss
                num_train_batches += 1

                batch_logs["loss"] = loss

            avg_train_loss = total_train_loss / num_train_batches
            epoch_logs["avg_train_loss"] = avg_train_loss

            if self.val_dataloader:
                self.evaluate(epoch, training_step)

    def evaluate(self, epoch: int, evaluation_step: Callable):
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with nw.no_grad():
            for batch_idx, batch_data in enumerate(self.val_dataloader):
                inputs, targets = batch_data[0], batch_data[1]
                outputs = self.model(inputs)
                loss = evaluation_step(self.model, inputs, targets, self.loss_fn)
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        val_logs = {"val_loss": avg_val_loss}

        self.model.train()

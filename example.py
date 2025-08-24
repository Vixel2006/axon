import nawah_api as nw
from nawah_api import Sequential, layers, loss, activations
from nawah_api import Optimizer, SGD
import numpy as np
from typing import Callable


class RegressionDataset(nw.Dataset):
    def __init__(self, num_samples=1000):
        self.X = (
            nw.Tensor.from_data(np.random.rand(num_samples, 1), requires_grad=True) * 10
        )  # Features

        self.y = self.X * 2 + 3

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


trainset = RegressionDataset(num_samples=10)
testset = RegressionDataset(num_samples=2)

trainloader = nw.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = nw.DataLoader(testset, batch_size=32, shuffle=True)

model = Sequential()
model.add("linear1", layers.linear(1, 1))

lr = 1e-2
optimizer = SGD(model.params.values(), lr=lr)
loss_fn = loss.mean_squared_error()["fn"]


def training_step(
    model: Sequential, inputs: nw.Tensor, targets: nw.Tensor, loss_fn: Callable
) -> nw.Tensor:
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    return loss


def evaluation_step(
    model: Sequential, inputs: nw.Tensor, targets: nw.Tensor, loss_fn: Callable
) -> nw.Tensor:
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    return loss


epochs = 100
log_interval = 10

print("\n--- Starting Training ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        loss = training_step(model, inputs, targets, loss_fn)
        loss.backward()
        optimizer.step()

        running_loss = loss + running_loss
        if (i + 1) % log_interval == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/log_interval}"
            )
            running_loss = 0.0

    # Evaluation after each epoch
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    for inputs, targets in testloader:
        loss = evaluation_step(model, inputs, targets, loss_fn)
        val_loss = loss + val_loss

    avg_val_loss = val_loss / len(testloader)
    print(
        f"Epoch [{epoch+1}/{epochs}] finished. Average Validation Loss: {avg_val_loss}"
    )

print("\n--- Training Finished ---")

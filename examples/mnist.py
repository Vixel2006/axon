import sys
from idrak.core.tensor import Tensor
from idrak.data.dataset import Dataset
from idrak.optim import SGD, Adam
import idrak.nn as nn
from idrak.nn.activations import Tanh, ReLU, Sigmoid, Softmax
from idrak.idrak_bindings.c_wrapper_functions import c_set_debug_mode
import idrak.metrics as metrics
import numpy as np
from mnist_datasets import MNISTLoader

if "DEBUG=1" in sys.argv:
    c_set_debug_mode(1)
    sys.argv.remove("DEBUG=1")
else:
    c_set_debug_mode(0)

# Define the dataset for the MNIST
class MNIST(Dataset):
    def __init__(self, train=True):
        loader = MNISTLoader()
        self.images, self.labels = loader.load(train=train)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Tensor(self.images[idx].shape, self.images[idx])
        # Convert label to one-hot encoding
        one_hot_label = np.eye(10)[self.labels[idx]].reshape(*self.labels[idx].shape, 10)
        label = Tensor(one_hot_label.shape, one_hot_label)
        return image, label

    def show(self, idx):
        img = self.__getitem__(idx)[0].data
        if img.ndim == 1:
            img = img.reshape(28, 28)
        img /= 255.0
        for row in img:
            line = "".join("█" if val > 0.5 else " " for val in row)
            print(line)

# Define the train and test sets
trainset = MNIST()
testset = MNIST(train=False)

# Define a naive dataloader
class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr < len(self.dataset):
            x = self.dataset[self.curr:self.curr+self.batch_size]
            self.curr += self.batch_size
            return x
        raise StopIteration


# Define the configuration for the model
class Config:
    BATCH_SIZE = 10
    EPOCHS = 1
    LR = 0.01
    IMSIZE = (28, 28)

# Define a datalaoder for the training loop
trainloader = DataLoader(trainset, Config.BATCH_SIZE)

# Define the model
model = nn.Sequential(
    nn.Linear(784, 512, bias=False),
    ReLU(),
    nn.Linear(512, 128, bias=False),
    ReLU(),
    nn.Linear(128, 10, bias=False),
    Softmax()
)

# Define the optimizer
optim = SGD(model.params, Config.LR)

# Now do the training
for epoch in range(Config.EPOCHS):
    for images, labels in trainloader:
        optim.zero_grad()
        pred = model(images)

        loss = metrics.bce(pred, labels)

        loss.backward()


        optim.step()


    print(f"Epoch [{epoch}/20]: Loss {loss}")




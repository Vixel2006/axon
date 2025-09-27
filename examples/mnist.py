from os import wait
import sys
from idrak.core.tensor import Tensor
from idrak.functions import *
from idrak.data.dataset import Dataset
from idrak.optim import SGD, Adam
import idrak.nn as nn
from idrak.nn.activations import Tanh, ReLU, Sigmoid, Softmax
import idrak.metrics as metrics
import numpy as np
from mnist_datasets import MNISTLoader


# Define the dataset for the MNIST
class MNIST(Dataset):
    def __init__(self, train=True):
        loader = MNISTLoader()
        images, labels = loader.load(train=train)
        
        # Filter for images of 0 and 1
        mask = (labels == 0) | (labels == 1)
        self.images = images[mask]
        self.labels = labels[mask]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            imgs = self.images[idx]
            labels = self.labels[idx]
            imgs = imgs.reshape(len(imgs), 784) / 255.0
            images = from_data(imgs.shape, imgs)
            labels = from_data((len(labels), 1), labels.reshape(-1,1))
            return images, labels
        else:
            img_data = self.images[idx].reshape(1, 784) / 255.0
            image = from_data(img_data.shape, img_data)
            label = from_data((1,1), [[self.labels[idx]]])
            return image, label


    def show(self, idx):
        img_data = self.images[idx]
        img_reshaped = img_data.reshape(28, 28)
        img_normalized = img_reshaped / 255.0
        for row in img_normalized:
            line = "".join("█" if val > 0.5 else " " for val in row)
            print(line)

# Define the train and test sets
trainset = MNIST()
testset = MNIST(train=False)

# Define a naive dataloader
class DataLoader:
    def __init__(self, dataset, batch_size):
        self.curr = 0
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
    BATCH_SIZE = 1
    EPOCHS = 5
    LR = 0.01
    IMSIZE = (28, 28)

# Define a datalaoder for the training loop
trainloader = DataLoader(trainset, Config.BATCH_SIZE)

# Define the model
model = nn.Sequential(
    nn.Linear(784, 1, bias=False),
    Sigmoid()
)

# Define the optimizer
optim = SGD(model.params, Config.LR)

for epoch in range(Config.EPOCHS):
    num_batches = 0
    for images, labels in trainloader:
        optim.zero_grad()
        pred = model(images)

        loss = metrics.bce(pred, labels)

        loss.backward()


        optim.step()

    print(f"Epoch [{epoch + 1}/{Config.EPOCHS}]: Loss {loss}")

# Evaluation
print("\n--- Evaluation ---")
correct_predictions = 0
for i in range(len(testset)):
    image, label = testset[i]

    # Show image
    print(f"Test sample #{i}")
    testset.show(i)
    label.realize()

    # Predict
    pred = model(image)
    pred.realize()
    
    predicted_label = 1 if pred.data[0, 0] > 0.5 else 0
    true_label = int(label.data[0,0])

    print(f"Predicted: {predicted_label}, True: {true_label}")
    if predicted_label == true_label:
        correct_predictions += 1
    print("-" * 28)

accuracy = (correct_predictions / len(testset)) * 100
print(f"\nEvaluation Accuracy: {accuracy:.2f}%")


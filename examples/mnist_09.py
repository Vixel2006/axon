import idrak.metrics as metrics
from idrak.functions import one_hot, from_data, squeeze
import os
import idrak.nn as nn
import idrak.experiment # Add this line
import numpy as np
from mnist_datasets import MNISTLoader
from idrak.data import Dataset, DataLoader
from idrak.nn.activations import Sigmoid, ReLU, Tanh, LogSoftmax
from idrak.optim import SGD, Adam
from idrak.utils.model_io import save_model, load_model # Import save_model and load_model


class MNIST(Dataset):
    def __init__(self, train=True):
        loader = MNISTLoader()
        images, labels = loader.load(train=train)
        
        if train:
            self.images = images[:1000]
            self.labels = labels[:1000]
        else:
            self.images = images[:100]
            self.labels = labels[:100]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            imgs = self.images[idx]
            labels = self.labels[idx]
            imgs = imgs.reshape(len(imgs), 784) / 255.0
            images = from_data(imgs.shape, imgs)
            labels = from_data((len(labels),), labels.reshape(-1,))
            return images, labels
        else:
            img_data = self.images[idx].reshape(1, 784) / 255.0
            image = from_data(img_data.shape, img_data)
            label = from_data((1,), [self.labels[idx]])
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

def run_experiment(experiment_id, experiment_name, description, model, optim, epochs):
    exp = idrak.experiment.Experiment(
        id=experiment_id,
        name=experiment_name,
        description=description
    )

    # Log hyperparameters
    exp.log_hyperparameter("batch_size", Config.BATCH_SIZE)
    exp.log_hyperparameter("epochs", epochs)
    exp.log_hyperparameter("learning_rate", Config.LR)
    
    # Dynamically get model architecture string
    model_architecture_str = " -> ".join([layer.__class__.__name__ for layer in model.layers])
    exp.log_hyperparameter("model_architecture", model_architecture_str)

    # Define a datalaoder for the training loop
    trainloader = DataLoader(trainset, Config.BATCH_SIZE)

    for epoch in range(epochs):
        num_batches = 0
        for images, labels in trainloader:
            optim.zero_grad()
            pred = model(images)
            
            labels = squeeze(labels, 1)
            labels_one_hot = one_hot(labels, 10)
            loss = metrics.nll_loss(pred, labels_one_hot)

            loss.backward()

            optim.step()

        print(f"Epoch [{epoch + 1}/{epochs}]: Loss {loss}")
        exp.log_metric("train_loss", loss.data[0], step=epoch + 1) # Log loss

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
        
        predicted_class = np.argmax(pred.data)
        true_class = int(label.data[0])

        if predicted_class == true_class:
            correct_predictions += 1

        print(f"Predicted: {predicted_class}, True: {true_class}")
        print("-" * 28)

    accuracy = correct_predictions / len(testset)
    print(f"\nEvaluation Accuracy: {accuracy}")
    exp.log_metric("test_accuracy", accuracy) # Log final test accuracy
    model_filepath = os.path.join("experiments", f"{experiment_id}_model.pkl")
    save_model(model, model_filepath)
    exp.log_artifact(model_filepath, name="trained_model")

    # Save experiment details
    exp.save()


# Define the configuration for the model
class Config:
    BATCH_SIZE = 50
    EPOCHS = 20
    LR = 0.01
    IMSIZE = (28, 28)

# First Experiment: Linear -> LogSoftmax
model_1 = nn.Pipeline(nn.Linear(784, 10, bias=False), LogSoftmax())
optim_1 = SGD(model_1.params, Config.LR)
run_experiment(
    experiment_id="mnist_classification_run_1",
    experiment_name="MNIST Classification - Linear + LogSoftmax",
    description="Training a simple linear model for MNIST digit classification with LogSoftmax and NLL loss.",
    model=model_1,
    optim=optim_1,
    epochs=Config.EPOCHS
)






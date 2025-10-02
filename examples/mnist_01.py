import idrak.metrics as metrics
from idrak.functions import *
import os
import idrak.nn as nn
import idrak.experiment # Add this line
import numpy as np
from mnist_datasets import MNISTLoader
from idrak.data import Dataset, DataLoader
from idrak.nn.activations import Sigmoid, ReLU, LogSoftmax # Add ReLU for future use
from idrak.optim import SGD, Adam
from idrak.utils.model_io import save_model, load_model # Import save_model and load_model


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

            loss = metrics.bce(pred, labels)

            loss.backward()


            optim.step()

        print(f"Epoch [{epoch + 1}/{epochs}]: Loss {loss}")
        exp.log_metric("train_loss", loss.data[0], step=epoch + 1) # Log loss

    # Evaluation
    #print("\n--- Evaluation ---")
    correct_predictions = 0
    for i in range(len(testset)):
        image, label = testset[i]

        # Show image
        #print(f"Test sample #{i}")
        #testset.show(i)
        label.realize()

        # Predict
        pred = model(image)
        pred.realize()
        
        predicted_label = 1 if pred.data[0, 0] > 0.5 else 0
        true_label = int(label.data[0,0])

        #print(f"Predicted: {predicted_label}, True: {true_label}")
        if predicted_label == true_label:
            correct_predictions += 1
        #print("-" * 28)
    accuracy = (correct_predictions / len(testset)) * 100
    print(f"\nEvaluation Accuracy: {accuracy:.2f}%")
    exp.log_metric("test_accuracy", accuracy) # Log final test accuracy

    # Save the model as an artifact
    model_filepath = os.path.join("experiments", f"{experiment_id}_model.pkl")
    save_model(model, model_filepath)
    exp.log_artifact(model_filepath, name="trained_model")

    # Save experiment details
    exp.save()


# Define the configuration for the model
class Config:
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 0.01
    IMSIZE = (28, 28)

# First Experiment: Linear -> Sigmoid
model_1 = nn.Pipeline(nn.Linear(784, 1, bias=False), Sigmoid())
optim_1 = Adam(model_1.params, Config.LR)
run_experiment(
    experiment_id="mnist_binary_classification_run_1",
    experiment_name="MNIST Binary Classification (0s and 1s) - Linear-Sigmoid",
    description="Training a simple linear model with Sigmoid for 0/1 MNIST classification.",
    model=model_1,
    optim=optim_1,
    epochs=Config.EPOCHS
)

print("\n" + "="*50 + "\n")
print("Starting Second Experiment with Modified Model")
print("\n" + "="*50 + "\n")

# Second Experiment: Modify model to Linear -> ReLU 
model_1[1] = ReLU()

# Here we reset the parameters of the model and optimizer so we start training from start, if you want to continue on the final state of the last experiement you can just comment these two lines 
model_1.reset_parameters()
optim_1 = Adam(model_1.params, Config.LR)

run_experiment(
    experiment_id="mnist_binary_classification_run_2",
    experiment_name="MNIST Binary Classification (0s and 1s) - Linear-ReLU-Linear-Sigmoid",
    description="Training a deeper model with ReLU activation for 0/1 MNIST classification.",
    model=model_1,
    optim=optim_1,
    epochs=Config.EPOCHS
)

print("\n" + "="*50 + "\n")
print("Demonstrating Model Loading and Prediction")
print("\n" + "="*50 + "\n")

# Load the model from the first experiment
loaded_model_path = os.path.join("experiments", "mnist_binary_classification_run_1_model.pkl")
loaded_model = load_model(loaded_model_path)

for i in range(5):
    # Predict on a random test image using the loaded model
    random_idx = np.random.randint(0, len(testset))
    image, label = testset[random_idx]

    print(f"Predicting for test sample #{random_idx}")
    testset.show(random_idx)
    label.realize()

    pred = loaded_model(image)
    pred.realize()

    predicted_label = 1 if pred.data[0, 0] > 0.5 else 0
    true_label = int(label.data[0,0])

    print(f"Loaded Model Predicted: {predicted_label}, True: {true_label}")

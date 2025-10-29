import os
import axon
import numpy as np
from axon import experiment
import axon.functions as F
import axon.metrics as metrics
from axon.data import Dataset, DataLoader
import axon.nn as nn
import axon.optim as optim
from axon.utils.model_io import save_model, load_model
from sklearn.datasets import fetch_openml

device = axon.Device("cuda")

# ========== Initializing the dataset for the training ===============
class Mnist(Dataset):
    def __init__(self, train: bool = True):
        print("Fetching MNIST Dataset with scikit-learn.")
        print("="*20)
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        y = y.astype(int)
        
        # Split data into training and testing sets (e.g., 60,000 for training, 10,000 for testing)
        x_train, x_test = X[:64], X[64:]
        y_train, y_test = y[:64], y[64:]

        if train:
            self.images = x_train
            self.labels = y_train
        else:
            self.images = x_test
            self.labels = y_test
        print("MNIST Dataset loaded successfully using scikit-learn.")
        print("="*20)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int | slice) -> tuple[axon.Tensor, axon.Tensor]:
        images_np, labels_np = self.images[idx].reshape(-1, 784), self.labels[idx]

        images_np = images_np / 255.0

        # One-hot encode the labels
        batch_size = images_np.shape[0]
        
        images = F.from_data((batch_size, 784), images_np)
        
        one_hot_labels_np = self.one_hot_encoding(batch_size, labels_np)
        
        # Convert one-hot encoded numpy array to axon.Tensor
        labels = F.from_data(one_hot_labels_np.shape, one_hot_labels_np)

        return images, labels

    def one_hot_encoding(self, batch_size: int, labels: np.ndarray) -> np.ndarray:
        num_classes = 10  # For MNIST, digits 0-9
        
        # Ensure labels is a 1D array, even if it's a single scalar
        labels = labels.flatten()

        # Create a zero array of shape (batch_size, num_classes)
        one_hot_labels = np.zeros((batch_size, num_classes), dtype=np.float32)
        
        # Set the appropriate elements to 1.0
        one_hot_labels[np.arange(batch_size), labels] = 1.0
        
        return one_hot_labels

    def display_digit(self, idx: int):
        try:
            img_data = self.images[idx]
            img_reshaped = img_data.reshape(28, 28)
            img_normalized = img_reshaped / 255.0
            for row in img_normalized:
                line = "".join("█" if val > 0.5 else " " for val in row)
                print(line)
        except Exception as e:
            print(f"Can't display the digit: {e}")

# Define our train and test sets
trainset = Mnist(train=True)
testset = Mnist(train=False)

# ======= Initialize the data for the first experiement ==========
class FFNExperiment:
    name = "Feed Forward Network Model for MNIST"
    description = "Using a Feed forward network model with 2 layer and adam optimizer for classifying hand-written digits"
    EPOCHS = 25
    BATCH_SIZE = 16
    LR = 0.01
    model = nn.Pipeline(nn.Linear(784, 10), nn.LogSoftmax()).to(device)
    optim = optim.Adam(model.params, lr=LR)

# ======== Define the dataloader for the data =====================
trainloader = DataLoader(trainset, batch_size=FFNExperiment.BATCH_SIZE)
testloader = DataLoader(testset, batch_size=1)

# ====== Define a function to run experiements ===========
def run_experiment():
    experiment = axon.Experiment(id="ua1893uae134", name=FFNExperiment.name, description=FFNExperiment.description)

    print(f"Starting experiment {FFNExperiment.name}")
    print("="*20)

    optim = FFNExperiment.optim
    model = FFNExperiment.model

    # Log the hyperparameters of the experiment
    hyperparams = {
        "epochs": FFNExperiment.EPOCHS,
        "batch_size": FFNExperiment.BATCH_SIZE,
        "lr": FFNExperiment.LR,
        "optimizer": "Adam"
    }

    for key, val in hyperparams.items():
        experiment.log_hyperparameter(key, val)

    # Start the training loop
    for epoch in range(FFNExperiment.EPOCHS):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            pred = model(images)

            loss = metrics.nll_loss(pred, labels)

            loss.backward()

            optim.step()


        experiment.log_metric("loss", loss.data[0], epoch)

        print(f"Epoch[{epoch+1}/{FFNExperiment.EPOCHS}]: Loss {loss.data[0]:.04}")

    # Save the model
    model_path = "models/ffn_mnist_model.pkl"
    save_model(model, model_path)

    # Save the model to the experiment
    experiment.log_artifact("model", model_path)

    # Save experiment
    experiment.save()

def evaluate_experiment(experiment_id):
    print(f"Evaluating the experiment {experiment_id}")
    print("="*20)

    # Load the experiment we want to evaluate using the id
    exp = axon.Experiment.load(experiment_id)

    # Getting the saved model from the experiment and load it (here the model is the only artifact so we can export it easily)
    model_path = exp.artifacts[0]['name']
    model = load_model(model_path)

    # Now we can use the model to do the evaluation and calculate the accuracy
    all_pred = 0
    true_pred = 0
    for i, (image, label) in enumerate(testset):
        pred = model(image)

        testset.display_digit(i)

        if np.argmax(pred.data) == np.argmax(label.data):
            true_pred += 1

        print(f"Prediction: {np.argmax(pred.data)}, Truth: {np.argmax(label.data)}")
        all_pred += 1

    print(f"Accuracy of the model: {(true_pred/all_pred) * 100}%")
    print("="*20)


if __name__ == "__main__":
    run_experiment()
    #evaluate_experiment("ua1893uae134")


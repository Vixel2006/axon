import nawah_api as nw
import time
import numpy as np
import os
import gzip
import struct
import requests

MNIST_URLS = {
    "train_images": "https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz",
    "train_labels": "https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz",
    "test_images": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(url, path):
    """Downloads a file from a URL to a given path."""
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            raise
    else:
        print(f"File already exists: {path}")


def load_idx_file(filepath):
    """Parses MNIST IDX binary files."""
    with gzip.open(filepath, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))

        if magic == 2051:  # Images file magic number
            num_rows, num_cols = struct.unpack(">II", f.read(8))
            print(f"Loading images: {num_items} items, {num_rows}x{num_cols}")
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_items, num_rows, num_cols
            )
        elif magic == 2049:  # Labels file magic number
            print(f"Loading labels: {num_items} items")
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic number: {magic} in {filepath}")
    return data


def get_mnist_data(data_dir="data"):
    """Downloads and loads MNIST data into NumPy arrays."""
    os.makedirs(data_dir, exist_ok=True)

    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    download_mnist(MNIST_URLS["train_images"], train_images_path)
    download_mnist(MNIST_URLS["train_labels"], train_labels_path)
    download_mnist(MNIST_URLS["test_images"], test_images_path)
    download_mnist(MNIST_URLS["test_labels"], test_labels_path)

    train_images = load_idx_file(train_images_path)
    train_labels = load_idx_file(train_labels_path)
    test_images = load_idx_file(test_images_path)
    test_labels = load_idx_file(test_labels_path)

    return train_images, train_labels, test_images, test_labels


# --- Prepare MNIST data for Nawah Tensors ---
def prepare_mnist_for_nawah(images_np, labels_np, device="cpu"):
    num_samples = images_np.shape[0]
    img_height, img_width = images_np.shape[1:]
    input_size = img_height * img_width

    nawah_image_tensors = []
    for i in range(num_samples):
        # Normalize and flatten image (28x28 -> 784)
        normalized_flattened_image = (
            (images_np[i].astype(np.float32) / 255.0).reshape(-1).tolist()
        )
        img_tensor = nw.Tensor(
            normalized_flattened_image, dtype=nw.DType.float32, device=device
        )

        img_tensor = img_tensor.unsqueeze(0)

        nawah_image_tensors.append(img_tensor)

    nawah_label_tensors = []
    for i in range(num_samples):
        # One-hot encode labels for 10 classes
        one_hot_label = np.zeros(10, dtype=np.float32)
        one_hot_label[labels_np[i]] = 1.0
        label_tensor = nw.Tensor(
            one_hot_label.tolist(), dtype=nw.DType.float32, device=device
        )

        label_tensor = label_tensor.unsqueeze(0)

        nawah_label_tensors.append(label_tensor)

    # TensorDataset will concatenate these lists of tensors
    # Resulting in a single (N, 784) tensor for images and (N, 10) for labels
    images_dataset = nw.TensorDataset(nawah_image_tensors)
    labels_dataset = nw.TensorDataset(nawah_label_tensors)

    return images_dataset, labels_dataset, input_size


# --- Training Configuration ---
DEVICE = "cpu"
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# --- Load Data ---
print(f"Loading MNIST data for training on {DEVICE.upper()}...")
train_images_np, train_labels_np, _, _ = get_mnist_data()

train_images_dataset, train_labels_dataset, input_feature_size = (
    prepare_mnist_for_nawah(train_images_np, train_labels_np, device=DEVICE)
)

num_train_samples = 60000
print(f"Training dataset size: {num_train_samples}")
print(f"Input feature size (flattened image): {input_feature_size}")

# --- Define the Model (Simple Fully Connected Network) ---
print("\nDefining the Neural Network Model...")
model = nw.Sequential()
model.add("fc1", nw.layers.linear(input_feature_size, 128))
model.add("relu1", nw.activations.relu())
model.add("fc2", nw.layers.linear(128, 64))
model.add("relu2", nw.activations.relu())
model.add("output_layer", nw.layers.linear(64, 10))

# Move the model to the specified device (CPU in this case)
model.to(DEVICE)

print("\nModel Summary:")
model.summary([1, input_feature_size])

print(f"\nStarting training on {DEVICE.upper()} for {EPOCHS} epochs...")
start_training_time = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    permutation = np.random.permutation(num_train_samples)

    for i in range(0, num_train_samples, BATCH_SIZE):
        batch_indices = permutation[i : i + BATCH_SIZE]

        X_batch_tensors = [train_images_dataset[j] for j in batch_indices]
        y_batch_tensors = [train_labels_dataset[j] for j in batch_indices]

        X_batch = nw.Tensor.cat(X_batch_tensors, dim=0)
        y_batch = nw.Tensor.cat(y_batch_tensors, dim=0)

        for param in model.params.values():
            param.zero_grad()

        predictions = model(X_batch)

        diff = predictions - y_batch
        loss = (diff * diff).sum()

        loss.backward()

        nw.SGD(model.params.values(), lr=LEARNING_RATE)

        epoch_loss += loss.data[0]

        pred_np = np.array(predictions.data).reshape(-1, 10)
        target_np = np.array(y_batch.data).reshape(-1, 10)

        predicted_classes = np.argmax(pred_np, axis=1)
        true_classes = np.argmax(target_np, axis=1)

        correct_predictions += np.sum(predicted_classes == true_classes)
        total_samples += len(batch_indices)

    avg_epoch_loss = epoch_loss / (num_train_samples / BATCH_SIZE)
    epoch_accuracy = correct_predictions / total_samples * 100

    print(
        f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
    )

end_training_time = time.time()
print(f"\nTraining finished in {end_training_time - start_training_time:.2f} seconds.")

print("\n--- Model Parameters After Training ---")
for name, param in model.params.items():
    print(
        f"Parameter: {name}, Shape: {param.shape}, Grad (partial): {param.grad[0] if param.grad and param.grad[0] else 'N/A'}"
    )

# --- Example of your original tests (adapted to CPU and current setup) ---
print("\n--- Original Test Snippets (CPU adaptation) ---")
a = nw.Tensor([[[1, 3, 4], [3, 4, 5], [3, 4, 5]]], device=DEVICE, requires_grad=True)
b = nw.Tensor([[[1, 3, 4], [3, 4, 5], [5, 6, 7]]], device=DEVICE, requires_grad=True)

"""
a = nw.Tensor([[[1, 3, 4], [3, 4, 5], [3, 4, 5]]], device="cpu", requires_grad=True)
b = nw.Tensor([[[1, 3, 4], [3, 4, 5], [5, 6, 7]]], device="cpu", requires_grad=True)

d = a.cat([a, b], dim=0)

print("This is concat")
print(d)

b.to("cuda:0")

c = a * b
print("---------------------")
print(c)
print("---------------------")
c.backward()

print("----------------------")
print(a.grad)
print("----------------------")
print("----------------------")
print(b.grad)
print("----------------------")

inp = nw.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], requires_grad=True, device="cpu")

kernel = nw.Tensor([[[1, 0], [0, -1]]], requires_grad=True, device="cpu")

print("Conv2d:")
print(nw.conv2d(inp, kernel))
print("-------------------------------------")

inpt = nw.Tensor([[1, 3, 4], [1, 3, 4]], requires_grad=True, device="cuda:0")

net = nw.Sequential()

net.add("conv2d_first", nw.layers.linear(3, 12))
net.add("relu1", nw.activations.relu())
net.add("fc1", nw.layers.linear(12, 1))

net.to("cuda:0")

net.summary([2, 3])

pred = net(inpt)

print("Prediction:")
print(pred)


print("Registered params")

truth = nw.ones([1, 1], requires_grad=True, device="cuda:0")


loss = truth - pred

loss.backward()

nw.SGD(net.params.values(), lr=0.1)
print("-------------------------------------------------")
print(net.params)

net1 = nw.Sequential(
    {
        "Conv2D": nw.layers.conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2)),
        "ReLU": nw.activations.relu(),
        "Flatten": nw.layers.flatten(),
        "FC1": nw.layers.linear(12, 1),
    }
)

net1.to("cuda:0")

net1.summary([3, 1, 3, 3])
"""

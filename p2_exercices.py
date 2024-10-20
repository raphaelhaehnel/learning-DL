import torch
from torch import nn
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

from helper_functions import plot_decision_boundary

RANDOM_SEED = 42

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_data(device, n_samples=1000):
    X, y = make_moons(n_samples=n_samples)

    # Convert the data to tensor
    X = torch.from_numpy(X).type(torch.float).to(device=device)
    y = torch.from_numpy(y).type(torch.float).to(device=device)

    return X, y

def training_loop(epochs: int, model: nn.Module, loss_fn, optimizer,
                  X_train, y_train, X_test, y_test, epoch_count,
                  train_loss_values, test_loss_values, device):

    accuracy_fn = Accuracy(task="binary").to(device)

    for epoch in range(epochs):

        model.train()

        # Raw output of our models
        y_logits = model(X_train).squeeze()

        # Prediction probabilities using the sigmoid function
        y_pred_probs = torch.sigmoid(y_logits)

        # Predicted labels
        y_pred = torch.round(y_pred_probs)

        loss = loss_fn(y_logits, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)

            acc = accuracy_fn(y_test, test_pred)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.cpu().detach().numpy())
            test_loss_values.append(test_loss.cpu().detach().numpy())

            print(f"Epoch: {epoch}, Loss: {loss:5f}, Acc: {acc*100:2f}%")


class MoonModel(nn.Module):
    def __init__(self, device: str):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(in_features=2, out_features=5, device=device),
                                   nn.ReLU(),
                                   nn.Linear(in_features=5, out_features=5, device=device),
                                   nn.ReLU(),
                                   nn.Linear(in_features=5, out_features=5, device=device),
                                   nn.ReLU(),
                                   nn.Linear(in_features=5, out_features=1, device=device),)

    def forward(self, x: torch.Tensor):
        return self.model(x)

def main_function():
    # Use GPU if available
    device = get_device()

    # Generate data
    X, y = get_data(device, n_samples=1000)

    # Split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Initialize the model
    model = MoonModel(device)

    # Define the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    # Define the number of epochs
    epochs = 500

    # Initialize different values to track
    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    training_loop(epochs, model, loss_fn, optimizer,
                      X_train, y_train, X_test, y_test, epoch_count,
                      train_loss_values, test_loss_values, device)

    plt.figure()
    plt.title("After training")
    plot_decision_boundary(model, X_test, y_test)

    plt.figure()
    plt.title(f"Loss function {model._get_name()}")
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.legend()

    plt.show()

def personalized_tanh_function():
    X = torch.linspace(-5, 5, steps=100)
    y = torch.tanh(X)

    manual_tanh = (torch.exp(X) - torch.exp(-X)) / (torch.exp(X) + torch.exp(-X))
    plt.plot(X, y, label="builtin tanh function")
    plt.plot(X, manual_tanh, label="My tanh")
    plt.legend()
    plt.show()

def spirals_data(N=100, K=3):
    # N: number of points per class
    # K: number of classes
    D = 2  # dimensionality
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # lets visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

if __name__ == "__main__":
    # main_function()
    # personalized_tanh_function()

    # Code for creating a spiral dataset from CS231n
    spirals_data(N=100, K=3)
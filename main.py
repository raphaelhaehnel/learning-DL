# If I need explainability -  DON'T USE DEEP LEARNING
# If the traditional approach is a better option - DON'T USE DEEP LEARNING
# If errors are unacceptable - DON'T USE DEEP LEARNING
# If I don't have much data - DON'T USE DEEP LEARNING
from cProfile import label

# Tensor datatype main errors:
# - Tensors not right datatype
# - Tensors not right shape
# - Tensors not on right device

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
PATH_MODEL = "./models/model.pt"

# Introduction to tensors
def intro_tensor():

    # Height, width, color channel
    rand_img = torch.rand(size=(1000, 1000, 3))
    plt.imshow(rand_img)
    plt.show()

def some_code():
    # intro_tensor()
    tensor_arranged = torch.arange(1, 10, step=2)
    print(tensor_arranged)
    print(torch.zeros_like(tensor_arranged,
                           dtype=None,
                           device=None,
                           requires_grad=False))

def conversions():
    np_array = np.array([1, 5])
    torch_array: torch.Tensor = torch.from_numpy(np_array)
    new_array = torch_array.numpy()
    print("end")

def define_gpu():
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("count devices: ", torch.cuda.device_count())
    return device

def prepare_and_load_data(device):
    X = torch.arange(0, 1, 0.02, device=device).unsqueeze(dim=1)
    weight = 0.7
    bias = 0.3

    y = weight * X + bias

    return X, y

def split_train_test(X, y):
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    return X_train, y_train, X_test, y_test

def plot_data(X_train, y_train, X_test, y_test, title):
    plt.figure()
    plt.scatter(X_train, y_train, c="b", label="Training data")
    plt.scatter(X_test, y_test, c="g", label="Testing data")
    plt.legend()
    plt.title(title)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1,
                                      bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def training_step(model, loss_fn, optimizer):

    # Set the model to training mode
    model.train()

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Gradient descent
    optimizer.step()

    # Turn off gradient tracking
    model.eval()

    return loss


def test_model(model, loss_fn):

    with torch.inference_mode():
        # 1. Do the forward pass
        test_pred = model(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

        return test_loss

def run_model(model):
    # Turn off gradient tracking. Like torch.no_grad but better
    with torch.inference_mode():
        y_pred = model(X_test)

    plot_data(X_train, y_train, X_test, y_pred, "Before training")

    # Current values of the parameters
    print(model.state_dict().values())

    # Setup a loss function
    loss_fn = nn.L1Loss()  # MAE function

    # Setup an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.01)  # stochastic gradient descent

    # Building a training loop
    epochs = 180

    # Track different values
    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    for epoch in range(epochs):
        loss = training_step(model, loss_fn, optimizer)

        test_loss = test_model(model, loss_fn)

        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())

    print(list(model.parameters()))

    with torch.inference_mode():
        test_pred = model(X_test)

    plot_data(X_train, y_train, X_test, test_pred, "After training")

    plt.figure()
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.legend()
    plt.title(f"Loss function {model._get_name()}")

    torch.save(model.state_dict(), PATH_MODEL)

if __name__ == '__main__':

    device = define_gpu()

    X, y = prepare_and_load_data(device)
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    # plot_data(X_train, y_train, X_test, y_test)

    torch.manual_seed(42)
    model_0 = LinearRegressionModel()
    model_1 = LinearRegressionModelV2()

    print(f"{model_0._get_name()} parameters: {model_0.state_dict()}")
    print(f"{model_1._get_name()} parameters: {model_1.state_dict()}")

    run_model(model_0)
    run_model(model_1)
    plt.show()


    print("end")
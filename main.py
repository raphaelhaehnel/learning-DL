# If I need explainability -  DON'T USE DEEP LEARNING
# If the traditional approach is a better option - DON'T USE DEEP LEARNING
# If errors are unacceptable - DON'T USE DEEP LEARNING
# If I don't have much data - DON'T USE DEEP LEARNING

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

def prepare_and_load_data():
    X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
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

if __name__ == '__main__':
    X, y = prepare_and_load_data()
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    # plot_data(X_train, y_train, X_test, y_test)

    torch.manual_seed(42)
    model_0 = LinearRegressionModel()

    # Turn off gradient tracking. Like torch.no_grad but better
    with torch.inference_mode():
        y_pred = model_0(X_test)

    plot_data(X_train, y_train, X_test, y_pred, "Before training")

    # Current values of the parameters
    print(model_0.state_dict().values())

    # Setup a loss function
    loss_fn = nn.L1Loss() # MAE function

    # Setup an optimizer
    optimizer = torch.optim.Adam(params=model_0.parameters(),
                                lr=0.01) # stochastic gradient descent

    # Building a training loop
    epochs = 150
    losses = []
    for epoch in range(epochs):

        # Set the model to training mode
        model_0.train()

        # model_0.eval() # turns off gradient tracking

        # 1. Forward pass
        y_pred = model_0(X_train)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Gradient descent
        optimizer.step()

        losses.append(loss.data.item())

    # Turn off gradient tracking
    model_0.eval()

    with torch.inference_mode():
        y_pred = model_0(X_test)

    plot_data(X_train, y_train, X_test, y_pred, "After training")

    plt.figure()
    plt.plot(losses)
    plt.title("Loss function")

    plt.show()
    print("end")
# Import pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import torchvision
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib
import matplotlib.pyplot as plt

from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm


def get_data(is_train: bool):
    data = datasets.FashionMNIST(root="data", # Where to download the data
                          train=is_train,
                          download=True,
                          transform=ToTensor(),
                          target_transform=None)

    return data

def display_sample_images(data, rows, cols):
    class_names = data.classes
    plt.figure()

    for i in range(1, rows * cols + 1):
        # Generate a random number
        random_idx = torch.randint(0, len(data), size=[1]).item()

        # Define in which subplot display the current image
        plt.subplot(rows, cols, i)

        # Get an image with its label from the train data
        image, label = data[random_idx]

        # Show the image in grayscale
        plt.imshow(image.squeeze(), cmap="gray")

        # Define a title to the image
        plt.title(f"{class_names[label]}")

        # Remove the axis from the graph
        plt.axis(False)

    plt.show()

def define_gpu():
    print("CUDA is available ? ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("count devices: ", torch.cuda.device_count())
    return device

def main():

    device = define_gpu()

    train_data = get_data(is_train=True)
    test_data = get_data(is_train=False)

    display_sample_images(train_data, rows=3, cols=6)

    # Set up the batch size hyperparameter
    BATCH_SIZE = 32

    # Turn datasets into iterables
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)


    class_names = train_data.classes

    model = FashionMNISTModelV1(input_shape=28*28,
                                  hidden_units=10,
                                  output_shape=len(class_names),
                                  device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                               lr=0.1)

    epochs = 20

    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_loss = 0

        for (batch, (X, y)) in enumerate(train_dataloader):

            X = X.to(device=device)
            y = y.to(device=device)

            model.train()

            # Forward passs
            y_pred = model(X)

            # Calculate the loss
            loss = loss_fn(y_pred, y)
            train_loss += loss

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()

            if batch % 400 == 0:
                print(f"Looked at {(batch * len(X))}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)

        # Testing
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:

                X_test = X_test.to(device=device)
                y_test = y_test.to(device=device)

                # Forward pass
                test_pred = model(X_test)

                # Calculate the loss
                test_loss += loss_fn(test_pred, y_test)

                # Calculate accuracy
                test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))

            # Calculate test loss average per batch
            test_loss /= len(test_dataloader)

            # Calculate the test acc average per batch
            test_acc /= len(test_dataloader)

        epoch_count.append(epoch)
        train_loss_values.append(train_loss.cpu().detach().numpy())
        test_loss_values.append(test_loss.cpu().detach().numpy())

        print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


    plt.figure()
    plt.title(f"Loss function {model._get_name()}")
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.legend()

    plt.show()

    model_0_results = eval_model(model,
                                 test_dataloader,
                                 loss_fn,
                                 accuracy_fn,
                                 device)

    print(model_0_results)
    print("end program")

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: str):
    """
    Returns a dictionary containing the results of model predicting on data_loader
    :param model:
    :param data_loader:
    :param loss_fn:
    :param accuracy_fn:
    :return:
    """

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)

            # Make predictions
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
        # Scale loss and acc to finc average loss,acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model._get_name(),
            "model_loss": loss.item(),
            "model_acc": acc}

def print_train_time(start:float,
                     end:float,
                     device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape))

    def forward(self, x):
        return self.layer_stack(x)

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, device: str):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units,
                      device=device),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape,
                      device=device))

    def forward(self, x):
        return self.layer_stack(x)

if __name__ == "__main__":
    main()
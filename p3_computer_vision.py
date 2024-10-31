# Import pytorch
import torch
import torchmetrics
from torch import nn, inference_mode
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
from tqdm import tqdm
import random
from mlxtend.plotting import plot_confusion_matrix

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

def train_model(device: str, model: nn.Module, test_dataloader, train_dataloader):

    # Setup loss function
    loss_fn = nn.CrossEntropyLoss()

    # Setup optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                               lr=0.1)

    epochs = 10

    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_loss = train_loop(device,
                                loss_fn,
                                model,
                                optimizer,
                                train_dataloader)

        # Testing
        test_acc, test_loss = testing_loop(device, loss_fn, model, test_dataloader)

        epoch_count.append(epoch)
        train_loss_values.append(train_loss.cpu().detach().numpy())
        test_loss_values.append(test_loss.cpu().detach().numpy())

        print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    display_loss_graph(epoch_count, model, test_loss_values, train_loss_values)

    model_0_results = eval_model(model,
                                 test_dataloader,
                                 loss_fn,
                                 accuracy_fn,
                                 device)

    print(model_0_results)

    torch.save(model.state_dict(), f"weights-{model._get_name()}.pt")

    print("end program")


def get_data_parameters():
    # Initialize the datasets
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

    # A list containing the class names
    class_names = train_data.classes

    return class_names, test_dataloader, train_dataloader, train_data, test_data


def display_loss_graph(epoch_count, model, test_loss_values, train_loss_values):
    plt.figure()
    plt.title(f"Loss function {model._get_name()}")
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.legend()
    plt.show()


def testing_loop(device, loss_fn, model, test_dataloader):
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
    return test_acc, test_loss


def train_loop(device: str,
               loss_fn: nn.Module,
               model: nn.Module,
               optimizer,
               train_dataloader):

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
    return train_loss


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


def make_predictions(model: nn.Module,
                     data: list,
                     device: str):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:

            # Prepare the sample
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob)

        return torch.stack(pred_probs)


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
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape,
                      device=device),
            nn.ReLU())

    def forward(self, x):
        return self.layer_stack(x)

class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, device: str):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape,
                      device=device)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def predict(model, test_data, class_names, device):
    test_samples = []
    test_labels = []
    pred_classes = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = make_predictions(model=model,
                                  data=test_samples,
                                  device=device)
    pred_classes = pred_probs.argmax(dim=1)

    plt.figure()
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i + 1)

        plt.imshow(sample.squeeze(), cmap="gray")

        pred_label = class_names[pred_classes[i]]

        truth_label = class_names[test_labels[i]]

        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")

        plt.axis(False)

if __name__ == "__main__":

    # Define running device
    device = define_gpu()

    # Get data
    class_names, test_dataloader, train_dataloader, train_data, test_data = get_data_parameters()

    # Define the model
    model_1 = FashionMNISTModelV1(input_shape=28 * 28,
                                hidden_units=10,
                                output_shape=len(class_names),
                                device=device)
    model_2 = FashionMNISTModelV2(input_shape=1,
                                hidden_units=10,
                                output_shape=len(class_names),
                                device=device)

    # Train model
    # train_model(device, model_2, test_dataloader, train_dataloader)
    model_2.load_state_dict(torch.load(f=f"weights-{model_2._get_name()}.pt"))

    # Padding: add a padding of 0 around the image, so the convolution is starting outside the image
    # Kernel size: size of the kernel. Number of iterations is (input size - kernel size + 1)
    # Stride: Steps of the kernel (usually 1)


    # Predict
    # predict(model_2, test_data, class_names, device)


    # 1. Make predictions with our trained model on the test dataset
    y_preds = []
    model_2.eval()
    with torch.inference_mode():
        for (X, y) in tqdm(test_dataloader, desc="Making predictions..."):

            X, y = X.to(device), y.to(device)
            y_logit = model_2(X)
            y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)

    # 2. Make a confusion matrix using torchmetrics.ConfusionMatrix
    confmat = torchmetrics.ConfusionMatrix(num_classes=len(class_names),
                                           task="multiclass")
    confmat_tensor = confmat(preds=y_pred_tensor,
                             target=test_data.targets)

    # 3. Plot the confusion matrix using mlxtend.plotting.plot_confusion_matrix()
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    class_names=class_names,
                                    figsize=(10, 7))



    plt.show()
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch


def get_device():
    """

    :return:
    """
    print("CUDA is available ? ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("count devices: ", torch.cuda.device_count())
    return device


def get_arguments_from_cmd():
    """

    :return:
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple example script")

    # Add arguments
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        help="Model name",
        required=True
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        help="Batch size",
        required=True
    )
    parser.add_argument(
        '-l', '--learning_rate',
        type=float,
        help="Learning rate",
        required=True
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        help="Num epochs",
        required=True
    )
    parser.add_argument(
        '-u', '--units',
        type=int,
        help="Hidden units",
        required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    model_name = str(args.model_name)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    num_epochs = int(args.epochs)
    hidden_units = int(args.units)

    print("Hyperparameters:")
    print(f"MODEL_NAME = {model_name}")
    print(f"BATCH_SIZE = {batch_size}")
    print(f"LEARNING_RATE = {learning_rate}")
    print(f"NUM_EPOCHS = {num_epochs}")
    print(f"HIDDEN_UNITS = {hidden_units}")

    return model_name, batch_size, learning_rate, num_epochs, hidden_units

def load_and_vectorize_data(path_data: str, batch_size: int, input_size: tuple[int, int], use_multicore: bool):
    """

    :param path_data:
    :param batch_size:
    :param input_size:
    :param use_multicore:
    :return:
    """

    # Define the transform
    data_transform = transforms.Compose([
        # Resize image
        transforms.Resize(size=input_size),
        # Flip the image randomly on horizontal
        transforms.RandomHorizontalFlip(p=0.5),
        # Turn the image into a torch tensor
        transforms.ToTensor()
    ])

    PATH_DATA_TRAIN = path_data + "train"
    PATH_DATA_TEST = path_data + "test"

    # Set up the datasets
    train_dataset = datasets.ImageFolder(root=PATH_DATA_TRAIN,
                                         transform=data_transform)
    test_dataset = datasets.ImageFolder(root=PATH_DATA_TEST,
                                        transform=data_transform)

    num_workers = os.cpu_count() if use_multicore else 0

    # Set up the dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True if use_multicore else False)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True if use_multicore else False)

    return train_dataset, train_dataloader, test_dataset, test_dataloader

def train_model(model: torch.nn,
                epochs: int,
                loss_fn: torch.nn.modules.loss,
                optimizer: torch.optim,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                device: str):
    """
    Train the model with the given data
    :param model: Model of the neural network to train
    :param epochs: The number of epochs
    :param loss_fn: The loss function
    :param optimizer: The optimizer
    :param train_dataloader: The Dataloader object containing the train dataset
    :param test_dataloader: The Dataloader object containing the test dataset
    :param device: The string representing the device, gpu or cpu
    :return: TODO
    """

    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    for epoch in range(epochs):
        print(f"Epoch {epoch} on {epochs}")

        # Initialize the train loss for the whole epoch
        train_loss = 0

        for (batch, (X, y)) in enumerate(train_dataloader):

            # Transfer the tensors to the defined device
            X: torch.Tensor = X.to(device)
            y: torch.Tensor = y.to(device)

            # Put the model in training mode
            model.train()

            # Forward pass
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
        test_acc, test_loss = testing_loop(device, loss_fn, model, test_dataloader)

        epoch_count.append(epoch)
        train_loss_values.append(train_loss.cpu().detach().numpy())
        test_loss_values.append(test_loss.cpu().detach().numpy())
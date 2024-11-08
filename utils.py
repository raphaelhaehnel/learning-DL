import argparse
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import torch

def get_device():
    print("CUDA is available ? ", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("count devices: ", torch.cuda.device_count())
    return device


def get_arguments_from_cmd():
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

    MODEL_NAME = str(args.model_name)
    BATCH_SIZE = int(args.batch_size)
    LEARNING_RATE = float(args.learning_rate)
    NUM_EPOCHS = int(args.epochs)
    HIDDEN_UNITS = int(args.units)

    print("Hyperparameters:")
    print(f"MODEL_NAME = {MODEL_NAME}")
    print(f"BATCH_SIZE = {BATCH_SIZE}")
    print(f"LEARNING_RATE = {LEARNING_RATE}")
    print(f"NUM_EPOCHS = {NUM_EPOCHS}")
    print(f"HIDDEN_UNITS = {HIDDEN_UNITS}")

    return MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, HIDDEN_UNITS

def load_and_vectorize_data(BATCH_SIZE: int):
    PATH_DATA = "data/pizza_steak_sushi/"
    PATH_DATA_TRAIN = PATH_DATA + "train"
    PATH_DATA_TEST = PATH_DATA + "test"

    train_dataset = datasets.ImageFolder(PATH_DATA_TRAIN)
    test_dataset = datasets.ImageFolder(PATH_DATA_TEST)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True) #TODO num_workers=os.cpu_count()

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False) #TODO num_workers=os.cpu_count()

    return train_dataset, train_dataloader, test_dataset, test_dataloader

def train_model(model: torch.nn):
    pass
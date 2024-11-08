import sys
import os

from nn_models.tiny_vgg import TinyVGG
from utils import *
from torchvision import transforms
import torch
from torch import nn

if __name__ == "__main__":

    # Get the current device to work on - cpu or gpu
    device = get_device()

    # Get the hyperparameters from the user via the cmd
    MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, HIDDEN_UNITS = get_arguments_from_cmd()

    # Load the data to datasets and dataloader
    train_dataset, train_dataloader, test_dataset, test_dataloader = load_and_vectorize_data(BATCH_SIZE)

    for (batch, (X, y)) in enumerate(train_dataloader):
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

    # Create our model
    model = TinyVGG(input_shape=3,
                    hidden_units=HIDDEN_UNITS,
                    output_shape=3,
                    device=device)

    # Load an existing model (if option is true)
    #TODO add condition
    model.load_state_dict(torch.load(f=f"models/weights-{model.__class__.__name__}.pt"))

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Setup an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=LEARNING_RATE)

    # Train our model
    train_model(model)

    # Save the model (if option is true)
    #TODO add condition
    torch.save(model.state_dict(), f=f"models/weights-{model.__class__.__name__}.pt")





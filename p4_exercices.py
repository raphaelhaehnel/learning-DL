from utils import *
from nn_models.tiny_vgg import TinyVGG
from torch import nn

PATH_DATA = "data/pizza_steak_sushi/"
LOAD_MODEL = False

if __name__ == "__main__":


    # Get the current device to work on - cpu or gpu
    device = get_device()

    # Get the hyperparameters from the user via the cmd
    MODEL_NAME, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, HIDDEN_UNITS = get_arguments_from_cmd()

    # Load the data to datasets and dataloader
    train_dataset, train_dataloader, test_dataset, test_dataloader = load_and_vectorize_data(path_data=PATH_DATA,
                                                                                             batch_size=BATCH_SIZE,
                                                                                             input_size=(64, 64),
                                                                                             use_multicore=False)
    # Get the number of channels (generally 3)
    number_of_channels = train_dataset[0][0].shape[0]

    # Create our model
    model = TinyVGG(input_shape=number_of_channels,
                    hidden_units=HIDDEN_UNITS,
                    output_shape=3,
                    device=device)

    # Load an existing model (if option is true)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(f=f"models/weights-{model.__class__.__name__}.pt"))

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Setup an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=LEARNING_RATE)

    # Train our model
    if not LOAD_MODEL:
        train_model(model, NUM_EPOCHS, loss_fn, optimizer, train_dataloader, test_dataloader, device)

    # Save the model (if option is true)
    if not LOAD_MODEL:
        torch.save(model.state_dict(), f=f"models/weights-{model.__class__.__name__}.pt")





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

def main():
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

    X_train_batch, y_train_batch = next(iter(train_dataloader))
    X_test_batch, y_test_batch = next(iter(test_dataloader))

    x = X_train_batch[0]

    # Create a flatten layer
    flatten_model = nn.Flatten()

    # Flatten the sample
    output = flatten_model(x)

    print("end program")

if __name__ == "__main__":
    main()
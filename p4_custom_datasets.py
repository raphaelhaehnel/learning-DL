from idlelib.pyparse import trans

import matplotlib.pyplot as plt
import torch
from numpy.ma.core import shape
from torch import nn
from pathlib import Path
import os
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict, List

from p3_computer_vision import define_gpu, train_model, create_confusion_matrix, predict

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = data_path / "pizza_steak_sushi" / "train"

# Get a list of paths of all the images
image_path_list = list(image_path.glob("*/*/*.jpg"))

random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem

img = Image.open(random_image_path)

plt.figure()
plt.imshow(img)
plt.axis(False)

data_transform = transforms.Compose([
    # Resize image
    transforms.Resize(size=(64, 64)),
    # Flip the image randomly on horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a torch tensor
    transforms.ToTensor()
])

img_transform: torch.Tensor = data_transform(img)

plt.figure()
plt.imshow(img_transform.permute(1, 2, 0))
# plt.imshow(img_transform.reshape((img_transform.shape[1], img_transform.shape[2], img_transform.shape[0])))
plt.axis(False)

# Create a dataset
train_data = datasets.ImageFolder(root="data/pizza_steak_sushi/train",
                                  transform=data_transform)
test_data = datasets.ImageFolder(root="data/pizza_steak_sushi/test",
                                  transform=data_transform)

# Turn the datasets into dataloader
train_dataloader = DataLoader(train_data,
                              batch_size=32,
                              shuffle=True)
test_dataloader = DataLoader(train_data,
                             batch_size=32,
                             shuffle=False)


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int,
                 device: str) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
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
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape,
                      device=device)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# We need to do data augmentation to increase the diversity of our dataset
# For example: rotate, crop, shift, etc...


# Define running device
device = define_gpu()

# Get classes names
class_names = train_data.classes

# Define the model
model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names),
                  device=device)

# Size tensors: [32, 3, 64, 64]

n_epochs = 200

# Train model
train_model(device, model_1, test_dataloader, train_dataloader, n_epochs)
# model_1.load_state_dict(torch.load(f=f"weights-{model_1._get_name()}.pt"))

# Predict
predict(model_1, test_data, class_names, device)

# Display confusion matrix
# create_confusion_matrix(model_1, test_dataloader, device)


plt.show()


from idlelib.pyparse import trans

import matplotlib.pyplot as plt
import torch
from numpy.ma.core import shape
from torch import nn
from pathlib import Path
import os
from PIL import Image
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = data_path / "pizza_steak_sushi" / "train"

# Get a list of paths of all the images
image_path_list = list(image_path.glob("*/*/*.jpg"))

random_image_path = random.choice(image_path_list)

image_class = random_image_path.parent.stem

img = Image.open(random_image_path)

# img.show()

# img = plt.imread(random_image_path)
# plt.imshow(img)
# plt.axis(False)
# plt.show()

# image_tensor = torch.from_numpy(img).reshape((img.shape[2], img.shape[0], img.shape[1]))
#
# print(f"tensor image: {image_tensor.shape}")
# print(f"array image: {img.shape}")

# print(image_tensor)

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
print(f"shape: {img_transform.shape}")

plt.figure()
plt.imshow(img_transform.permute(1, 2, 0))
# plt.imshow(img_transform.reshape((img_transform.shape[1], img_transform.shape[2], img_transform.shape[0])))
plt.axis(False)


train_data = datasets.ImageFolder(root="data/pizza_steak_sushi/train",
                                  transform=data_transform)
test_data = datasets.ImageFolder(root="data/pizza_steak_sushi/test",
                                  transform=data_transform)

train_dataloader = DataLoader(train_data,
                              batch_size=32,
                              shuffle=True,
                              num_workers=os.cpu_count())
test_dataloader = DataLoader(train_data,
                             batch_size=32,
                             shuffle=False,
                             num_workers=os.cpu_count())
plt.show()
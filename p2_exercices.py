import torch
from mpl_toolkits.axisartist.angle_helper import select_step24
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
RANDOM_SEED = 42

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_data():
    return make_moons(n_samples=1000, random_state=RANDOM_SEED)

class MoonModel(nn.Module):
    def __init__(self, device: str):
        super().__init__(self)

        nn.Sequential() #TODO we are here
        
    def forward(self):
        pass

device = get_device()
X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


plt.figure()
plt.scatter(x=X[:, 0], y=X[:, 1], s=2, c=y)
plt.show()
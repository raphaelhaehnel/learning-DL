from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)


circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             s=2,
#             cmap=plt.set_cmap("RdYlBu"))
# plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Build a model
class CircleModelV0(nn.Module):
    def __init__(self, device: str):
        super().__init__()

        # Use nn.Linear() for creating the model parameters
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=5,
                                 bias=True,
                                 device=device)
        self.layer_2 = nn.Linear(in_features=5,
                                 out_features=1,
                                 bias=True,
                                 device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x))

device = "cpu"
model_0 = CircleModelV0(device)

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

print("end")

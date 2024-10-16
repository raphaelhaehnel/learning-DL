from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import requests
from pathlib import Path

# Download helper function from Learn PyToch repo
if Path("helper_functions.py").is_file():
    print("helper_functions.py already existsm skipping download")
else:
    print("Downloading helper_function.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

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
    untrained_preds = model_0(X_test)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct/len(y_pred) * 100
    return acc

# Train model

epochs = 100

for epoch in range(epochs):

    model_0.train()

    # Raw output of our models
    y_logits = model_0(X_train).squeeze()

    # Prediction probabilities using the sigmoid function
    y_pred_probs = torch.sigmoid(y_logits)

    # Predicted labels
    y_pred = torch.round(y_pred_probs)

    # Calculate loss
    loss = loss_fn(y_logits, y_train)

    # Calculate accuracy
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward (backpropagation)
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()

        test_pred = torch.round(torch.sigmoid((test_logits)))

        test_loss = loss_fn(test_logits, y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    print(f"Epoch: {epoch}, Loss: {loss:5f}, Acc: {acc:2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

plt.show()
print("end")

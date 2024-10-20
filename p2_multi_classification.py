from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import requests
from pathlib import Path
from torchmetrics import Accuracy

from p1_regression import define_gpu

# Download helper function from Learn PyToch repo
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_function.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

device = define_gpu()

# Make 1000 samples
n_samples = 1000

n_clusters = 4

# Create circles
X, y = make_blobs(n_samples=n_samples,
                  n_features=2,
                  centers=n_clusters,
                  cluster_std=1.5,
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

X = torch.from_numpy(X).type(torch.float).to(device=device)
y = torch.from_numpy(y).type(torch.long).to(device=device)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)



class BlobModelV1(nn.Module):
    def __init__(self, device: str):
        super().__init__()

        # Use nn.Linear() for creating the model parameters
        self.layer_1 = nn.Linear(in_features=2,
                                 out_features=4,
                                 bias=True,
                                 device=device)
        self.layer_2 = nn.Linear(in_features=4,
                                 out_features=5,
                                 bias=True,
                                 device=device)
        self.layer_3 = nn.Linear(in_features=5,
                                 out_features=n_clusters,
                                 bias=True,
                                 device=device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3((self.layer_2((self.layer_1(x)))))



model = BlobModelV1(device)

with torch.inference_mode():
    untrained_preds = model(X_test)

loss_fn = nn.CrossEntropyLoss()
torchmetrics_acc = Accuracy(task="multiclass", num_classes=n_clusters).to(device)

optimizer = torch.optim.Adam(params=model.parameters(),
                            lr=0.01)

# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct/len(y_pred) * 100
    return acc

# Train model

epochs = 1000

# Track different values
epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):

    model.train()

    # Raw output of our models
    y_logits = model(X_train).squeeze()

    # Prediction probabilities using the sigmoid function
    y_pred_probs = torch.softmax(y_logits, dim=1)

    # Predicted labels
    y_pred = torch.argmax(y_pred_probs, dim=1)

    # Calculate loss
    loss = loss_fn(y_logits, y_train)

    # Calculate accuracy
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    torch_acc = torchmetrics_acc(y_pred, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backward (backpropagation)
    loss.backward()

    # Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()

        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    if epoch % 100 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.cpu().detach().numpy())
        test_loss_values.append(test_loss.cpu().detach().numpy())

        print(f"Epoch: {epoch}, Loss: {loss:5f}, Acc: {acc:2f}%, Torch-acc: {torch_acc*100:2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)

plt.figure()
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.legend()
plt.title(f"Loss function {model._get_name()}")

plt.show()
print("end")

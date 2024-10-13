import torch
from main import PATH_MODEL
from main import LinearRegressionModel

# Create the model
model_0 = LinearRegressionModel()

# Load the model's parameters
model_0.load_state_dict(torch.load(f=PATH_MODEL))
with torch.inference_mode():
    pass

print("end")
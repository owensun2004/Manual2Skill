import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, input_dim):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # First layer
        self.fc2 = nn.Linear(512, 128)   # Second layer
        self.fc3 = nn.Linear(128, 9)    # Third layer
        self.relu = nn.ReLU()           # Activation function

    def forward(self, x):
        x1 = self.relu(self.fc1(x))   # First layer + ReLU
        x2 = self.relu(self.fc2(x1))   # Second layer + ReLU
        x3 = self.fc3(x2)              # Third layer (no activation)
        return x3
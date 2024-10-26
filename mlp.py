import torch
import torch.nn as nn

# define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        out = self.fc1(x)  # First fully connected layer
        out = self.relu(out)  # ReLU activation
        out = self.fc2(out)  # Second fully connected layer
        out = self.relu(out)  # ReLU activation (if needed)
        out = self.fc3(out)  # Output layer
        return out
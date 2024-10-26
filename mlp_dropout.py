import torch
import torch.nn as nn

# Define the MLP model with Dropout
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout layer with specified probability

    def forward(self, x):
        out = self.fc1(x)  # First fully connected layer
        out = self.relu(out)  # ReLU activation
        out = self.dropout(out)  # Apply dropout after the first layer

        out = self.fc2(out)  # Second fully connected layer
        out = self.relu(out)  # ReLU activation
        out = self.dropout(out)  # Apply dropout after the second layer

        out = self.fc3(out)  # Output layer (no dropout here)
        return out
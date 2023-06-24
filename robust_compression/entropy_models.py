import torch
import torch.nn as nn


class NullEntropyModel(nn.Module):
    def __init__(self):
        super(NullEntropyModel, self).__init__()


    def forward(self, x):
        return x

# Define your entropy model architecture
class EntropyModel(nn.Module):
    def __init__(self, compressed_size):
        super(EntropyModel, self).__init__()
        self.fc = nn.Linear(compressed_size, compressed_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        probabilities = self.softmax(self.fc(x))
        return probabilities
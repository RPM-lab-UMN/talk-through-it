import torch
import torch.nn as nn
from torchvision import models

class L2A(nn.Module):
    def __init__(self, h1 = 512):
        super().__init__()

        # create MLP layers
        self.fc1 = nn.Linear(h1, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, prev_action, command):
        # concatenate x and y
        x = torch.cat((prev_action, command), dim=1)
        # pass through MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # create model
    model = L2A()
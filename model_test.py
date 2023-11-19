import torch.nn as nn
import torch.nn.functional as F
import torch


class ManNatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv4 = nn.Conv2d(24, 48, 3, padding=1)
        self.fc1 = nn.Linear(48 * 8 * 8, 64)  
        # self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
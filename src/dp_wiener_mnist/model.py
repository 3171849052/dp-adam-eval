"""Original DP-KFC SimpleCNN architecture and initialization order."""

import torch
from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        img_size: int = 28,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        fc_size = img_size // 4
        self.fc1 = nn.Linear(32 * fc_size * fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

import torch
from torch import nn


def conv2d(in_channels: int,
           out_channels: int,
           kernel_size: int = 3,
           stride: int = 1,
           activation: nn.Module = nn.ReLU,
           padding=None,
           ) -> nn.Sequential:
    if padding is None and stride == 1:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        activation()
    )


class ConvNet(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.layer1 = conv2d(1, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        return self.fc(x)

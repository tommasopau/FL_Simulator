import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    A base model class that other models can inherit from.
    """
    def __init__(self):
        super().__init__()


class MNISTCNN(BaseModel):
    """
    MNIST Convolutional Neural Network model.

    Architecture:
        - conv1: Convolutional layer with 1 input channel, 32 output channels, 3x3 kernel, stride 1, padding 1
        - conv2: Convolutional layer with 32 input channels, 64 output channels, 3x3 kernel, stride 1, padding 1
        - pool: Max pooling layer with 2x2 window
        - fc1: Fully connected layer with 7*7*64 input units and 128 output units
        - fc2: Fully connected layer with 128 input units and 10 output units for classification
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass of the MNISTCNN model.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FCMNIST(BaseModel):
    """
    Fully Connected MNIST model.

    Architecture:
        - fc1: Fully connected layer with 784 input units and 512 output units
        - fc2: Fully connected layer with 512 input units and 10 output units for classification
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """
        Forward pass of the FCMNIST model.
        """
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ZalandoCNN(BaseModel):
    """
    Zalando Convolutional Neural Network model.

    Architecture:
        - layer1: Sequential container with Conv2d, BatchNorm2d, ReLU, and MaxPool2d layers
        - layer2: Sequential container with Conv2d, BatchNorm2d, ReLU, and MaxPool2d layers
        - fc1: Fully connected layer with 64*6*6 input units and 600 output units
        - drop: Dropout layer with 25% dropout rate
        - fc2: Fully connected layer with 600 input units and 120 output units
        - fc3: Fully connected layer with 120 input units and 10 output units for classification
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        """
        Forward pass of the ZalandoCNN model.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
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


class FNet(BaseModel):
    def __init__(self):
        super(FNet,self).__init__()
        self.fc1=nn.Linear(784,512)
        self.fc2=nn.Linear(512,256)
        self.out=nn.Linear(256,10)
        
        # Dropout probability - set for avoiding overfitting
        self.dropout=nn.Dropout(0.2)

    def forward(self,x):
        x = x.view(-1, 28 * 28)        
        x=self.dropout(F.relu(self.fc1(x)))
        x=self.dropout(F.relu(self.fc2(x)))
        x=self.out(x)
        return x


class ResNetCIFAR10(BaseModel):
    """
    ResNet model for the CIFAR-10 dataset.
    Uses a modified ResNet-18 from torchvision:
      - Changes the first convolution layer to accept 3-channel inputs.
      - Removes the initial maxpool layer to better handle small images.
      - Adjusts the final fully connected layer for 10 output classes.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=False)
        # Modify the first convolution layer for CIFAR10 (3 channels, smaller images)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool layer to retain more spatial information for CIFAR10
        self.model.maxpool = nn.Identity()
        # Adjust the final fully connected layer for 10-class classification
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class AdultCensusIncomeClassifier(BaseModel):
    """
    Neural network for the Adult Census Income classification task.

    Architecture:
      - fc1: Fully connected layer mapping input_dim to 64 hidden units
      - fc2: Fully connected layer mapping 64 units to 32 hidden units
      - fc3: Fully connected layer mapping 32 units to num_classes outputs (default 2 for binary classification)
      - Dropout is applied after fc1 and fc2 to help prevent overfitting.
    """
    def __init__(self, input_dim = 14, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CovertypeClassifier(BaseModel):
    """
    Neural network for the Forest Covertype classification task.

    Architecture:
      - fc1: Fully connected layer mapping 54 input features to 128 hidden units
      - fc2: Fully connected layer mapping 128 hidden units to 64 hidden units
      - fc3: Fully connected layer mapping 64 hidden units to 7 output classes
      - Dropout is applied after fc1 and fc2 to help prevent overfitting.
    """
    def __init__(self, input_dim=54, num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class KDDSimpleNN(BaseModel):
    """
    Neural network model for the KDD Cup 1999 dataset.

    Architecture:
      - fc1: Fully connected layer mapping input_dim to 128 hidden units
      - fc2: Fully connected layer mapping 128 hidden units to 64
      - fc3: Fully connected layer mapping 64 hidden units to 32
      - fc4: Fully connected layer mapping 32 hidden units to num_classes outputs
      - Dropout is applied after fc1 and fc2 to help prevent overfitting.

    Note:
      Ensure that the input_dim reflects the actual number of features after any necessary preprocessing.
    """
    def __init__(self, input_dim = 41, num_classes = 23):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
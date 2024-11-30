import torch.nn as nn
import torch.nn.functional as F
from torch import load as torch_load


class VegetableCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(VegetableCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(num_classes=13):
    return VegetableCNN(num_classes)


def load_model(model, load_path='models/trained_model.pth'):
    model.load_state_dict(torch_load(load_path, weights_only=True))
    print(f"Modelo carregado de: {load_path}")
    return model

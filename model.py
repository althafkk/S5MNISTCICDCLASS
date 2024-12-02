import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 28, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(28)
        self.dropout1 = nn.Dropout(0.02)
        
        self.conv3 = nn.Conv2d(28, 12, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(12)
        self.dropout2 = nn.Dropout(0.02)
        
        self.fc1 = nn.Linear(12 * 3 * 3, 72)
        self.bn4 = nn.BatchNorm1d(72)
        self.fc2 = nn.Linear(72, 10)
        
    def forward(self, x):
        identity = x
        x1 = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x1)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 4)
        x = self.dropout2(x)
        
        x = x.view(-1, 12 * 3 * 3)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x 
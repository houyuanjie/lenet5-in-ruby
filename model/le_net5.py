import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        last = self.conv1(input)
        last = F.relu(last)
        last = F.max_pool2d(last, kernel_size=2)

        last = self.conv2(last)
        last = F.relu(last)
        last = F.max_pool2d(last, kernel_size=2)

        last = self.flatten(last)

        last = self.fc1(last)
        last = F.relu(last)

        last = self.fc2(last)
        last = F.relu(last)

        return self.fc3(last)

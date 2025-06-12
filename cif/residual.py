import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):

    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels,channels,
                               kernel_size=3,padding=1)
        self.con2 = nn.Conv2d(channels,channels,
                              kernel_size=3,padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.con2(y)
        return F.relu(x + y)
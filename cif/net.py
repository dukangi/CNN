import torch
from torch import nn

from cif.residual import ResidualBlock


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3,32,5,1,2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32,32,5,1,2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.rblock2 = ResidualBlock(32)
        self.conv3 = nn.Conv2d(32,64,5,1,2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.rblock3 = ResidualBlock(64)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64*4*4,64)
        self.linear2 = nn.Linear(64,10)
        #可写可写，因为CrossEntropyLoss自带了softmax
       #self.softmax = nn.softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.rblock1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.rblock2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.rblock3(x)

        x = self.flatten(x)
        x = self.linear1(x)

        x = self.linear2(x)

        return x


if __name__ == '__main__':
    x = torch.rand(1,3,32,32)
    model = MyModel()
    out = model(x)
    print(out)
import torch
from torch import nn
from torch.autograd import Variable


class NIN(nn.Module):

    def __init__(self, in_channels=3, out_channels=10):
        super(NIN, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5)
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5)
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(192, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.gap = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return x


if __name__ == '__main__':
    nin = NIN()
    x = Variable(torch.randn(4, 3, 32, 32))
    print(nin(x).size())
    print(nin)

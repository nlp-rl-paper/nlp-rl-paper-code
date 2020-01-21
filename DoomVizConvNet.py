from torch import nn
import torch.nn.functional as F
import torch


class VizConvNet(nn.Module):
    def __init__(self,n_channels,hidden_units ,available_actions_count,resolution,device):
        super(VizConvNet, self).__init__()
        self.device = device
        self.channels = n_channels
        self.dim0 = ((((resolution[0] - 6 )// 3 + 1 ) - 3 )// 2 + 1)
        self.dim1 = ((((resolution[1] - 3 )// 3 + 1 ) - 3 )// 2 + 1)
        self.conv1 = nn.Conv2d(n_channels, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(self.dim1*self.dim0*16, hidden_units)
        self.fc2 = nn.Linear(hidden_units, available_actions_count)

    def forward(self, x):
        x.to(self.device)
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
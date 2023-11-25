'''
Build MNISTModels

 - simple MLP with 3 layers
 - simple CNN model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    
    def __init__(self, input_dim=28*28, hdims=[128, 64], 
            out_dim=10):

        super(SimpleMLP,self).__init__()

        net = [] 
        prev_layer = input_dim 

        for hdim in hdims:
            net.append(nn.Linear(in_features=prev_layer,
                out_features=hdim, dtype=torch.float32))
            net.append(nn.ReLU())
            prev_layer = hdim

        net.append(nn.Linear(in_features=hdims[-1],
            out_features=10, dtype=torch.float32))
        self.net = nn.ModuleList(net)

    def init_params(self):
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class SimpleCNN(nn.Module):

    def __init__(self, input_dim=(28,28), input_channels=1, 
            kernel_size=[3,3], hconv_channels=[16, 32], 
            stride=1, hdims=[1024, 256]):

        super(SimpleCNN, self).__init__()

        # Conv modules
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=hconv_channels[0],
            kernel_size=kernel_size, 
            padding=(kernel_size[0]-1)//2)

        self.pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=hconv_channels[0], 
            out_channels=hconv_channels[1],
            kernel_size=kernel_size,
            padding=(kernel_size[0]-1)//2)

        self.pool2 = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        # FC Layers
        self.lin1 = nn.Linear(7*7*hconv_channels[1], hdims[0], 
            dtype=torch.float32)
        self.lin2 = nn.Linear(hdims[0], hdims[1], 
            dtype=torch.float32)
        self.lin3 = nn.Linear(hdims[1], 10, dtype=torch.float32)

    def init_params(self):
        
        for layer in (self.conv1, self.conv2, self.lin1, 
                self.lin2, self.lin3):

            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.lin1(x.view((x.size(0), -1))))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x

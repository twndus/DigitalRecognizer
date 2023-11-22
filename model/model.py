'''
Build MNISTModels

 - simple MLP with 3 layers
 - simple CNN model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    
    def __init__(self):
        super(SimpleMLP,self).__init__()
        self.layer1 = nn.Linear(in_features=784, out_features=100, dtype=torch.float32)
        self.layer2 = nn.Linear(in_features=100, out_features=50, dtype=torch.float32)
        self.layer3 = nn.Linear(in_features=50, out_features=10, dtype=torch.float32)

    def init_params(self):
        nn.init.kaiming_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        nn.init.kaiming_normal_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x

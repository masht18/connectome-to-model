import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class ClassifierReadout(nn.Module):
    def __init__(self, output_size, n_classes, intermediate_dim=100):
        super(ClassifierReadout, self).__init__()
        
        h, w, dim = output_size
        self.fc1 = nn.Linear(h * w * dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, n_classes)
        
    def forward(self, x):
        pred = self.fc1(F.relu(torch.flatten(x, start_dim=1)))
        pred = self.fc2(F.relu(pred))
        return pred
    

class ActionReadout(nn.Module):
    def __init__(self, output_size, act_space_sz):
        super(ActionReadout, self).__init__()
        
        h, w, dim = output_size
        self.fc = nn.Linear(h * w * dim, act_space_sz*2)
        
    def forward(self, x):
        params = self.fc(F.relu(torch.flatten(x, start_dim=1)))
        dists = Normal(*params)
        return dists
    
class ValueReadout(nn.Module):
    def __init__(self, output_size):
        super(ActionReadout, self).__init__()
        
        h, w, dim = output_size
        self.fc = nn.Linear(h * w * dim, 1)
        
    def forward(self, x):
        value = self.fc(F.relu(torch.flatten(x, start_dim=1)))
        return value
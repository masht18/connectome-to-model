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
        
    def forward(self, x, eval_mode=False):
        pred = self.fc1(F.relu(torch.flatten(x, start_dim=1)))
        if eval_mode:
            return pred
        pred = self.fc2(F.relu(pred))
        return pred
    
class ThalamusMLP(nn.Module):
    '''
    Takes multiple outputs from brain-like graph and mixes them. 
    '''
    def __init__(self, input_sizes, output_dim, h1=128, h2=64):
        super(ActionReadout, self).__init__()
        
        num_outputs = len(input_sizes)
        self.modules = []
        
        for i in input_sizes:
            h, w, dim = input_sizes[i]
            mlp = nn.Sequential(nn.Linear(h * w * dim, h1),
                                nn.ReLU(),
                               nn.Linear(h1, h2))
            self.modules.append(mlp)
            
        self.final_fc = nn.Linear(h2*num_outputs, output_dim)
        
    def forward(self, x):
        
        return Normal(mean, std)

class ActionReadout(nn.Module):
    def __init__(self, output_size, act_dim, h_dim=100):
        super(ActionReadout, self).__init__()
        
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        h, w, dim = output_size
        self.fc1 = nn.Linear(h * w * dim, h_dim)
        self.fc2 = nn.Linear(h_dim, act_dim)
        
    def forward(self, x):
        mean = self.fc1(F.relu(torch.flatten(x, start_dim=1)))
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
class ValueReadout(nn.Module):
    def __init__(self, output_size, h_dim=100):
        super(ValueReadout, self).__init__()
        
        h, w, dim = output_size
        self.fc1 = nn.Linear(h * w * dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        
    def forward(self, x):
        value = self.fc1(F.relu(torch.flatten(x, start_dim=1)))
        value = self.fc2(F.relu(value))
        return value
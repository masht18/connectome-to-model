import torch
from torch import nn
import torch.nn.functional as F

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
    
class MultiOutputClassifierReadout(nn.Module):
    def __init__(self, output1_size, output2_size, n_classes, intermediate_dim=100):
        super(MultiOutputClassifierReadout, self).__init__()
        
        h1, w1, dim1 = output1_size
        h2, w2, dim2 = output2_size
        self.fc1 = nn.Linear(h1*w1*dim1 + h2*w2*dim2 +1, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, n_classes)
        
    def forward(self, x1, x2, align_flag):
        align_flag = torch.unsqueeze(align_flag, 1)
        x = (torch.flatten(x1, start_dim=1), torch.flatten(x2, start_dim=1), align_flag)
        x = torch.cat(x, dim=1)
        pred = self.fc1(F.relu(x))
        pred = self.fc2(F.relu(pred))
        return pred
    
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
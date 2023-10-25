import numpy as np
import os
from torch import log
import torch.nn
import torch.optim
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

class Agent():
    '''The agent class that is to be filled.
       You are allowed to add any method you
       want to this class.
    '''

    def __init__(self, obs_space, act_space, graph, batch_size):
        
        # environment
        self.obs_space_sz = obs_space.shape[0]
        
        # Model
        self.model = Architecture(graph, input_sizes, input_dims,
                    topdown=args['topdown']).cuda().float()
        self.actor = ActionReadout(self.model.output_size, act_space)
        self.critic = ValueReadout(self.model.output_size)
        

    def act(self, obs):
        h = self.model(obs)
        dist = self.actor(h)
        a = dist.mean()
        
        return a
        
    def step(self, obs):
        h = self.model(obs)
        dist = self.actor(h)
        v = self.critic(h).flatten()
        a = dist.sample()
        
        return a, v, dist.log_prob(a)
    
    def pi(self, obs, acts=None):
        h = self.model(obs)
        dist = self.actor(h)
        
        return dist
        
    def v(self, obs):
        h = self.model(obs)
        v = self.critic(h).flatten()
        
        return v
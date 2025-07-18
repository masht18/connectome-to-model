'''
Classic MNIST handwritten digit identification
Basic task for demonstrating basic functionality only. 
'''

import torch
import math
import pickle
import argparse
import os

from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T

import gymnasium as gym

from model.graph import Graph, Architecture

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 50)
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--reps', type = int, default = 1, help = 'how many times to view input')
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar/convgru_feedback/graphs/test/topdown_test_mult_only.csv')
parser.add_argument('--data_path', type = str, default = '/home/mila/m/mashbayar/datasets')

parser.add_argument('--model_save', type = str, default = 'topdowna.pt')
parser.add_argument('--results_save', type = str, default = 'topdown_only_noupsampling_mult.npy')

args = vars(parser.parse_args())

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(args['seed'])
print(device)

#-------------------------------------------------
## 1: Create environment
env = gym.make('CartPole-v1')
observation, info = env.reset()

#---------------------------------------------------
## 2: Load graph, choose input and output areas

input_nodes = [0]   # Which areas do inputs go to, can be multiple
output_node = 1     # Which area do you get output from, must be a single area
graph_loc = args['graph_loc']
graph = Graph(graph_loc, input_nodes=input_nodes, output_node=output_node)

input_sizes = graph.find_input_sizes()    # Expected height, width of input for each area
input_dims = graph.find_input_dims()      # Expected channel dimensions of input for each area 

#-----------------------------------------------------
## 3: Initialize neural network
model = Agent(graph, observation, )

# Optimizer & loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

def test_sequence(dataloader, clean_data):
    
    '''
    Inference
        :param dataloader
            dataloader to draw the target image from
        :param clean data (torchvision.Dataset)
            clean dataset to draw bottom-up sequence images from
    '''
    correct = 0
    total = 0

    with torch.no_grad():

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            imgs, label = data
            imgs, label = imgs.to(device), label.to(device)
                
            input_list = [torch.unsqueeze(imgs, 1)]

            output = model(input_list)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return correct/total

def train_sequence():
    running_loss = 0.0
        
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
            
        imgs, label = data
        imgs, label = imgs.to(device), label.to(device)
        
        input_list = [torch.unsqueeze(imgs, 1)]
        
        output = model(input_list)
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss

losses = {'loss': [], 'train_acc': [], 'val_acc': []}    

if os.path.exists(args['model_save']):
    model.load_state_dict(torch.load(args['model_save']))
    print("Loading existing model")
else:
    print("No pretrained model found. Training new one.")

for epoch in range(args['epochs']):
    train_acc = test_sequence(train_loader, train_data)
    val_acc = test_sequence(test_loader, test_data)
    loss = train_sequence()
    
    printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |val accuracy {:1.5f}'.format(epoch, loss, train_acc, val_acc)

    print(printlog)
    losses['loss'].append(loss)
    losses['train_acc'].append(train_acc)
    losses['val_acc'].append(val_acc)

    with open(args['results_save'], "wb") as f:
        pickle.dump(losses, f)

    torch.save(model.state_dict(), args['model_save'])

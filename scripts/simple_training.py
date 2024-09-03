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

from connectome_to_model.utils.datagen import *
from connectome_to_model.model.graph import Graph, Architecture
from connectome_to_model.model.readouts import ClassifierReadout

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def test_sequence(dataloader):
    
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
            imgs, label = torch.unsqueeze(imgs, 1).to(device), label.to(device)
                
            y = model([imgs])
            output = readout(y[0])

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return correct/total

def train_sequence():
    running_loss = 0.0
        
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
            
        imgs, label = data
        imgs, label = torch.unsqueeze(imgs, 1).to(device), label.to(device)
        
        y = model([imgs])
        output = readout(y[0])
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
    parser.add_argument('--epochs', type = int, default = 50)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--reps', type = int, default = 1, help = 'how many times to view input')
    parser.add_argument('--topdown', type = str2bool, default = True)
    parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/test/topdown_test_mult_only.csv')
    parser.add_argument('--data_path', type = str, default = '/home/mila/m/mashbayar.tugsbayar/datasets')

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
    ## 1: Prepare dataset
    print('Loading datasets')

    transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    MNIST_path='/home/mila/m/mashbayar.tugsbayar/datasets'
    train_data = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=transform)
    test_data = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=transform)

    trainloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=32)
    testloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=32)

    #---------------------------------------------------
    ## 2: Load graph, choose input and output areas
    
    input_nodes = [0]              # Which areas do inputs go to, can be multiple
    output_node = 2                # Which area do you get output from
    graph_loc = args['graph_loc']
    graph = Graph(graph_loc, input_nodes=input_nodes, output_nodes=output_node)
    
    input_sizes = graph.find_input_sizes()    # Expected height, width of input for each area
    input_dims = graph.find_input_dims()      # Expected channel dimensions of input for each area 

    #-----------------------------------------------------
    ## 3: Initialize neural network
    model = Architecture(graph, input_sizes, input_dims).cuda().float()
    readout = ClassifierReadout(model.output_sizes[0], n_classes=10).cuda().float()

    params = [{'params': model.parameters(), 'lr': 0.001},
              {'params': readout.parameters(), 'lr': 0.001}]

    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()

    losses = {'loss': [], 'train_acc': [], 'match_acc': []}    

    if os.path.exists(args['model_save']):
        model.load_state_dict(torch.load(args['model_save']))
        print("Loading existing ConvGRU model")
    else:
        print("No pretrained model found. Training new one.")

    for epoch in range(args['epochs']):
        train_acc = test_sequence(trainloader)
        match_acc = test_sequence(testloader)
        loss = train_sequence()

        printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |match accuracy {:1.5f}'.format(epoch, loss, train_acc, match_acc)

        print(printlog)
        losses['loss'].append(loss)
        losses['train_acc'].append(train_acc)
        losses['match_acc'].append(match_acc)

        with open(args['results_save'], "wb") as f:
            pickle.dump(losses, f)

        torch.save(model.state_dict(), args['model_save'])
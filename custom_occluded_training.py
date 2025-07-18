'''
Pared-down training script for testing basic functionality only
'''

import torch
import math
import pickle
import argparse


from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils.datagen import *
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
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/mnist/semibio_thicc.csv')
#parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--model_save', type = str, default = 'saved_models/occluded_mult_semibio.pt')
parser.add_argument('--results_save', type = str, default = 'results/topdown_tests/mult_semibio.npy')

args = vars(parser.parse_args())

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

# # 1: Prepare dataset
print('Loading datasets')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the Occluded MNIST dataset from a .pt file

test_data = torch.load('occluded_mnist_single_test_32.pt')
train_data = torch.load('occluded_mnist_single_train_32.pt')

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=1)

criterion = nn.CrossEntropyLoss()

# INIT GRAPH
input_node = [0] # A
output_node = 3 #C
input_dims = [1, 0, 0, 0]
input_sizes = [(32, 32), (0, 0), (0, 0), (0,0)]
graph_loc = args['graph_loc']
graph = Graph(graph_loc, input_nodes=input_node, output_node=output_node)

# INIT MODEL
model = Architecture(graph, input_sizes, input_dims,
                    topdown=args['topdown']).cuda().float()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def test_sequence(dataloader):
    
    '''
    Inference
        :param dataloader
            dataloader to draw the target image from
    '''
    correct = 0
    total = 0

    with torch.no_grad():

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            imgs, label = data
            imgs = imgs[0]
            imgs, label = imgs.to(device), label.to(device)
            #print(imgs.shape)
                
            
            input_list = []
            imgs = torch.unsqueeze(imgs, 1)
            input_list.append(imgs)
            #input_list.append(topdown)


            output = model(input_list)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    
    return correct/total

def train_sequence():
    running_loss = 0.0
        
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
            
        imgs, label = data
        imgs = imgs[0]
        imgs, label = imgs.to(device), label.to(device)

        #input_seqs = sequence_gen(imgs, label, train_data, mnist_ref_train, seq_style='addition').float()

        # Generate random topdown for testing purposes only
        #topdown = torch.rand(imgs.shape[0], input_seqs.shape[1], args['topdown_c'], args['topdown_h'], args['topdown_w']).to(device)
        
        input_list = []
        imgs = torch.unsqueeze(imgs, 1)
        input_list.append(imgs)
        #input_list.append(input_seqs)
        #input_list.append(topdown)
        
        output = model(input_list)
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss

losses = {'loss': [], 'train_acc': [], 'val_acc': []}    

if os.path.exists(args['model_save']):
    model.load_state_dict(torch.load(args['model_save']))
    print("Loading existing ConvGRU model")
else:
    print("No pretrained model found. Training new one.")

for epoch in range(args['epochs']):
    train_acc = test_sequence(train_loader)
    val_acc = test_sequence(test_loader)
    loss = train_sequence()
    
    printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |val accuracy {:1.5f}'.format(epoch, loss, train_acc, val_acc)

    print(printlog)
    losses['loss'].append(loss)
    losses['train_acc'].append(train_acc)
    losses['val_acc'].append(val_acc)

    with open(args['results_save'], "wb") as f:
        pickle.dump(losses, f)

    torch.save(model.state_dict(), args['model_save'])
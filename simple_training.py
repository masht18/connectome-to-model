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
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils.datagen import *
from model.graph import Graph, Architecture

from utils.oscar_utils import StereoImageFolder

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
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--topdown_type', type = str, default = 'multiplicative')
parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/topdown_test_decreasing.csv')
#parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--model_save', type = str, default = 'saved_models/topdown_test_dec_comp.pt')
parser.add_argument('--results_save', type = str, default = 'results/topdown_tests/topdown_only_upsamplingILC_comp.npy')

args = vars(parser.parse_args())

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

# # 1: Prepare dataset
print('Loading datasets')

transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
MNIST_path='/home/mila/m/mashbayar.tugsbayar/datasets'
train_data = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=transform)
test_data = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=transform)

# Label references, which help generate sequences (these guys tell you where to look in the dataset if you need a 6, 4, etc)
mnist_ref_train = generate_label_reference(train_data)
mnist_ref_test = generate_label_reference(test_data)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()

#connections = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
#node_params = [(1, 28, 28, 5, 5), (10, 15, 15, 5, 5), (10, 9, 9, 3, 3), (10, 3, 3, 3, 3)]

# INIT GRAPH
input_nodes = [0, 2] # V1
output_node = 1 #IT
input_dims = [1, 0, 1]
input_sizes = [(32, 32), (0, 0), (32, 32)]
graph_loc = args['graph_loc']
graph = Graph(graph_loc, input_nodes=input_nodes, output_node=output_node)

# INIT MODEL
model = Architecture(graph, input_sizes, input_dims,
                    topdown=args['topdown'],
                    topdown_type=args['topdown_type']).cuda().float()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

def test_sequence(dataloader, clean_data, dataset_ref):
    
    '''
    Inference
        :param dataloader
            dataloader to draw the target image from
        :param clean data (torchvision.Dataset)
            clean dataset to draw bottom-up sequence images from
        :param dataset_ref (list)
            if providing image clue, provide label reference as well
    '''
    correct = 0
    total = 0

    with torch.no_grad():

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            imgs, label = data
            imgs, label = imgs.to(device), label.to(device)
                
            # Generate a sequence that adds up to loaded image    
            input_seqs = sequence_gen(imgs, label, clean_data, dataset_ref, seq_style='addition').float()
            # Generate random topdown
            #topdown = torch.rand(imgs.shape[0], input_seqs.shape[1], args['topdown_c'], args['topdown_h'], args['topdown_w']).to(device)
            
            input_list = [torch.unsqueeze(input_seqs[:, 0], 1), torch.unsqueeze(input_seqs[:, 2], 1)]
            #imgs = torch.unsqueeze(imgs, 1)
            #input_list.append(imgs)
            #input_list.append(topdown)
            #input_list.append(input_seqs)


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
        imgs, label = imgs.to(device), label.to(device)

        input_seqs = sequence_gen(imgs, label, train_data, mnist_ref_train, seq_style='addition').float()

        # Generate random topdown for testing purposes only
        #topdown = torch.rand(imgs.shape[0], input_seqs.shape[1], args['topdown_c'], args['topdown_h'], args['topdown_w']).to(device)
        
        input_list = [torch.unsqueeze(input_seqs[:, 0], 1), torch.unsqueeze(input_seqs[:, 2], 1)]
        #imgs = torch.unsqueeze(imgs, 1)
        #input_list.append(imgs)
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
    train_acc = test_sequence(train_loader, train_data, mnist_ref_train)
    val_acc = test_sequence(test_loader, test_data, mnist_ref_test)
    loss = train_sequence()
    
    printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |val accuracy {:1.5f}'.format(epoch, loss, train_acc, val_acc)

    print(printlog)
    losses['loss'].append(loss)
    losses['train_acc'].append(train_acc)
    losses['val_acc'].append(val_acc)

    with open(args['results_save'], "wb") as f:
        pickle.dump(losses, f)

    torch.save(model.state_dict(), args['model_save'])

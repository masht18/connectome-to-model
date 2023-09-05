import torch
import math
import pickle
import argparse
import random
import os
import hub

from ambiguous.dataset.dataset import DatasetTriplet

from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data_utils

from utils.datagen import *
from model.graph import Graph, Architecture
from utils.audio_dataset import MELDataset
from utils.audio_dataset import AudioVisualDataset

def fetch_clues(labels, reference_imgs=None, dataset_ref=None):
    clues = torch.zeros((labels.shape[0], 1, 28, 28)) #dummy function to produce the right size
        
    return clues.to(device)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 20)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--topdown_type', type = str, default = 'audio')
parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/multimodal_random3.csv')
parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--model_save', type = str, default = 'saved_models/audio_wmismatch_brainlike_random3.pt')
parser.add_argument('--results_save', type = str, default = 'results/audio_recovery_brainlike/random3.npy')

args = vars(parser.parse_args())

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(args['seed'])
print(device)

# %% [markdown]
# # 1: Prepare dataset
print('Loading datasets')

# Create dataloaders
#transform = T.Resize((32, 32))
#amb_trainset = DatasetTriplet('/network/scratch/m/mashbayar.tugsbayar/datasets/amnistV5', split='train', transform=transform)
#amb_testset = DatasetTriplet('/network/scratch/m/mashbayar.tugsbayar/datasets/amnistV5', split='test', transform=transform)
#amb_trainloader = DataLoader(amb_trainset, batch_size=64, shuffle=True)
#amb_testloader = DataLoader(amb_testset, batch_size=64, shuffle=True)

# FSDD dataset in case of audio topdown
#audio_ds = hub.load("hub://activeloop/spoken_mnist")
#mel_ds = MELDataset(audio_ds)
#audio_ref = generate_label_reference(audio_ds, dataset_type='fsdd')


def load_mixed_dataset(split, *root):
    datasets = [AudioVisualDataset(None, None, cache_dir=cache_dir, split=split) for cache_dir in root]
    combined = ConcatDataset(datasets)
    return combined

amb_match_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_match'
clean_mismatch_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'
amb_mismatch_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_mismatch'
audio_only_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_audio_only'
    
trainset = load_mixed_dataset('train', amb_match_root, clean_mismatch_root)
#trainset = AudioVisualDataset(None, None, cache_dir=audio_only_root, split='train')
match_testset = AudioVisualDataset(None, None, cache_dir=amb_match_root, split='test')
mismatch_testset = AudioVisualDataset(None, None, cache_dir=clean_mismatch_root, split='test')
control_testset = AudioVisualDataset(None, None, cache_dir=amb_mismatch_root, split='test')
#control_testset = AudioVisualDataset(None, None, cache_dir=audio_only_root, split='test')
    
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
match_testloader = DataLoader(match_testset, batch_size=32, shuffle=False, num_workers=2)
mismatch_testloader = DataLoader(mismatch_testset, batch_size=32, shuffle=False, num_workers=2) 
control_testloader = DataLoader(control_testset, batch_size=32, shuffle=False, num_workers=2) 

def test_sequence(dataloader, align_to='image'):
    total = 0
    correct = 0
    label_idx = 1 if align_to == 'audio' else 0
        
    for i, data in enumerate(dataloader, 0):
            x, label = data
            label = label[label_idx].to(device)
            x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]
            
            output = model(x)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    
    return correct/total

def train_sequence(align_to='image'):
    running_loss = 0
    test_acc = []
    label_idx = 1 if align_to == 'audio' else 0
        
    for i, data in enumerate(trainloader, 0):
            
        optimizer.zero_grad()
            
        x, label = data
        label = label[label_idx].to(device)
        x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]
            
        output = model(x)
            
        loss = criterion(output, label)
        running_loss += loss.item()
        #test_acc.append(test_sequence(amb_testloader))
            
        loss.backward()
        optimizer.step()
    
    return running_loss, test_acc

criterion = nn.CrossEntropyLoss()
# INIT GRAPH
#input_nodes = [0, 2] # V1
#output_node = 1 #IT
#input_dims = [1, 0, 1]
#input_sizes = [(32, 32), (0, 0), (32, 32)]
input_nodes = [0, 4] # V1, A1
output_node = 3 #IT
graph_loc = args['graph_loc']
graph = Graph(graph_loc, input_nodes=input_nodes, output_node=output_node)
input_sizes = graph.find_input_sizes()
input_dims = graph.find_input_dims()
input_dims[4] = 1
print(input_dims)

# INIT MODEL
model = Architecture(graph, input_sizes, input_dims,
                    topdown=args['topdown']).cuda().float()

optimizer = optim.Adam(model.parameters(), lr=0.001)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

losses = {'loss': [], 'train_acc': [], 'amb_match_acc': [], 'clean_mismatch_acc': [], 'ambimg_acc': [], 'ambimg_audio_align': []}    

if os.path.exists(args['model_save']):
    model.load_state_dict(torch.load(args['model_save']))
    print("Loading existing ConvGRU model")
else:
    print("No pretrained model found. Training new one.")

for epoch in range(args['epochs']):
    train_acc = test_sequence(trainloader)
    match_acc = test_sequence(match_testloader)
    mismatch_acc = test_sequence(mismatch_testloader)
    ambimg_acc = test_sequence(control_testloader, align_to='image')
    ambimg_audio_alignment = test_sequence(control_testloader, align_to='audio')
    loss, test_acc = train_sequence()
    
    printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |match accuracy {:1.5f} | mismatch accuracy {:1.5f} |control image alignment {:1.5f} |control audio alignment {:1.5f}'.format(epoch, loss, train_acc, match_acc, mismatch_acc, ambimg_acc, ambimg_audio_alignment)

    print(printlog)
    losses['loss'].append(loss)
    losses['train_acc'].append(train_acc)
    losses['amb_match_acc'].append(match_acc)
    losses['clean_mismatch_acc'].append(mismatch_acc)
    losses['ambimg_acc'].append(ambimg_acc)
    losses['ambimg_audio_align'].append(ambimg_audio_alignment)

    with open(args['results_save'], "wb") as f:
        pickle.dump(losses, f)

    torch.save(model.state_dict(), args['model_save'])
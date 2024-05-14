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
from torchvision.transforms import Compose

from connectome_to_model.utils.datagen import *
from connectome_to_model.utils.mech_eval import *
from connectome_to_model.model.graph import Graph, Architecture
from connectome_to_model.model.readouts import ClassifierReadout
from connectome_to_model.utils.audio_dataset import MELDataset
from connectome_to_model.utils.audio_dataset import AudioVisualDataset

def fetch_clues(labels, reference_imgs=None, dataset_ref=None):
    clues = torch.zeros((labels.shape[0], 1, 28, 28)) #dummy function to produce the right size
        
    return clues.to(device)

def randomize(y):
    return random.randint(0,9)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def load_mixed_dataset(split, *root):
    datasets = [AudioVisualDataset(None, None, cache_dir=cache_dir, split=split, transforms=T.Resize((32,32)),target_transforms=t_transforms) for cache_dir in root]
    #print(datasets[0][0][0][0].shape)
    #print(datasets[0][0][1][0].shape)
    #print(datasets[0][0][0][1].shape)
    #print(datasets[0][0][1][1].shape)
    combined = ConcatDataset(datasets)
    return combined

def test_sequence(dataloader, align_to='image'):
    total = 0
    correct = 0
    label_idx = 1 if align_to == 'audio' else 0
        
    with torch.no_grad():    
        for i, data in enumerate(dataloader, 0):
            x, label = data
            label = label[label_idx].to(device)
            x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]

            y = model(x)
            #y = model([x[1]])
            output = readout(y[0])

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return correct/total

def train_sequence(align_to='image'):
    running_loss = 0
    label_idx = 1 if align_to == 'audio' else 0
        
    for i, data in enumerate(trainloader, 0):
            
        optimizer.zero_grad()
            
        x, label = data
        label = label[label_idx].to(device)
        x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]
            
        y = model(x)
        #y = model([x[1]])
        output = readout(y[0])
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--reps', type = int, default = 1)
    parser.add_argument('--batch_sz', type = int, default = 32)
    parser.add_argument('--audio_ambiguity', type = str2bool, default = True)
    parser.add_argument('--img_ambiguity', type = str2bool, default = True)
    parser.add_argument('--match', type = str2bool, default = True)
    parser.add_argument('--align_to', type = str, default = 'audio')
    parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/ambaudio/multimodal_brainlike_MPC.csv')
    parser.add_argument('--connection_decay', type = str, default = 'ones')
    parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

    parser.add_argument('--model_save', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/saved_models/brainlike_doubleamb_MPC2.pt')
    parser.add_argument('--results_save', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/results/brainlike_doubleamb_MPC2.npy')

    args = vars(parser.parse_args())

    # %%
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed for reproducibility
    torch.manual_seed(args['seed'])
    #torch.set_num_threads(1)
    #print(device)
    t_transforms = lambda y: torch.tensor(y).to(device)
    
    # Roots of different multimodal datasets. 
    # 1st letter audio: ambiguous (a) or unambiguous (u)
    # 2nd letter img, 3rd letter: match (m) or mismatch (n)
    
    aam_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aam'
    aan_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aan'
    
    aum_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aum'
    aun_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aunv'
    
    #uam_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uam'
    uam_root = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_match'
    #uan_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uan'
    uan_root = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_mismatch'
    
    uum_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uum'
    #uun_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'
    uun_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uun'

    # %% [markdown]
    # # 1: Prepare dataset
    print('Loading datasets')
    
    # Load Audio
    #trainset = AudioVisualDataset(None, None, uun_root, split='train')
    trainset = load_mixed_dataset('train', aum_root, uun_root)
    testset = AudioVisualDataset(None, None, aum_root, split='test',transforms=T.Resize((32,32)))
    mismatch_testset = AudioVisualDataset(None, None, uun_root, split='test',transforms=T.Resize((32,32)), target_transforms=t_transforms)
    control_testset = AudioVisualDataset(None, None, aun_root, split='test', transforms=T.Resize((32,32)), target_transforms=t_transforms)
    allamb_control = AudioVisualDataset(None, None, aan_root, split='test', transforms=T.Resize((32,32)), target_transforms=t_transforms)
    
    trainloader = DataLoader(trainset, batch_size=args['batch_sz'], shuffle=True)
    testloader = DataLoader(testset, batch_size=args['batch_sz'], shuffle=True)
    mismatch_testloader = DataLoader(mismatch_testset, batch_size=args['batch_sz'], shuffle=True)
    control_testloader = DataLoader(control_testset, batch_size=args['batch_sz'], shuffle=True)
    allamb_control_testloader = DataLoader(allamb_control, batch_size=args['batch_sz'], shuffle=True)

    # INIT GRAPH
    input_nodes = [0, 4] # V1, A1
    output_node = 3 if args['align_to'] == 'image' else 6
    graph_loc = args['graph_loc']
    graph = Graph(graph_loc, input_nodes=input_nodes, output_nodes=output_node)
    input_sizes = graph.find_input_sizes()
    input_dims = graph.find_input_dims()

    # INIT MODEL
    model = Architecture(graph, input_sizes, input_dims).cuda().float()
    readout = ClassifierReadout(model.output_sizes[0], n_classes=10).cuda().float()

    params = [{'params': model.parameters(), 'lr': 0.001},
              {'params': readout.parameters(), 'lr': 0.001}]

    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()
    print(model)

    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in model_parameters])
    #print(params)

    losses = {'loss': [], 'train_acc': [], 'amb_match_acc': [], 'clean_mismatch_acc': [], 
              'control_img_align': [], 'control_audio_align': [],
             'allamb_img_align': [], 'allamb_audio_align': []}    

    if os.path.exists(args['model_save']):
        model.load_state_dict(torch.load(args['model_save']))
        print("Loading existing ConvGRU model")
    else:
        print("No pretrained model found. Training new one.")

    for epoch in range(args['epochs']):
        #train_acc = test_sequence(trainloader, align_to=args['align_to'])
        train_acc = 0
        match_acc = test_sequence(testloader, align_to=args['align_to'])
        #match_acc = test_sequence(mismatch_testloader, align_to='image')
        mismatch_acc = test_sequence(mismatch_testloader, align_to=args['align_to'])
        control_img_align = test_sequence(control_testloader, align_to='image')
        control_audio_align = test_sequence(control_testloader, align_to='audio')
        umap = dim_reduction(model, control_testset, datapoints=400)
        with open('dim_red/audio/dim_red_tsne_MPC_control.npy', "wb") as f:
            pickle.dump(umap, f)
        #allamb_img_align = test_sequence(allamb_control_testloader, align_to='image')
        #allamb_audio_align = test_sequence(allamb_control_testloader, align_to='audio')
        allamb_img_align = 0
        allamb_audio_align = 0
        #loss = train_sequence(align_to=args['align_to'])
        loss = 0

        printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |match accuracy {:1.5f} | mismatch accuracy {:1.5f} |control image alignment {:1.5f} |control audio alignment {:1.5f} |allamb image alignment {:1.5f} |allamb audio alignment {:1.5f}'.format(epoch, loss, train_acc, match_acc, mismatch_acc, control_img_align, control_audio_align, allamb_img_align, allamb_audio_align)

        #print(printlog)
        losses['loss'].append(loss)
        losses['train_acc'].append(train_acc)
        losses['amb_match_acc'].append(match_acc)
        losses['clean_mismatch_acc'].append(mismatch_acc)
        losses['control_img_align'].append(control_img_align)
        losses['control_audio_align'].append(control_audio_align)
        losses['allamb_img_align'].append(allamb_img_align)
        losses['allamb_audio_align'].append(allamb_audio_align)

        with open(args['results_save'], "wb") as f:
            pickle.dump(losses, f)

        torch.save(model.state_dict(), args['model_save'])

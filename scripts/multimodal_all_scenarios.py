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
from connectome_to_model.utils.mech_eval import fetch_hstates
from connectome_to_model.model.graph import Graph, Architecture
from connectome_to_model.model.readouts import ClassifierReadout
from connectome_to_model.utils.audio_dataset import MELDataset
from connectome_to_model.utils.audio_dataset import AudioVisualDataset

def randomize(y):
    return random.randint(0,9)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def fetch_latents_all_tasks(model, areas=8, timesteps=7):
    h_states = torch.zeros(4, areas, timesteps)
    
    vs_testloader = DataLoader(vs_testset, batch_size=400, shuffle=True)
    as_testloader = DataLoader(as_testset, batch_size=400, shuffle=True)
    mismatch_testloader = DataLoader(mismatch_testset, batch_size=400, shuffle=True)
    
    h_states[0] = fetch_hstates(model, vs_testloader, align_to='image', attention_flag=True)
    h_states[1] = fetch_hstates(model, as_testloader, align_to='audio', attention_flag=True)
    h_states[2] = fetch_hstates(model, mismatch_testloader, align_to='image', attention_flag=True)
    h_states[3] = fetch_hstates(model, mismatch_testloader, align_to='audio', attention_flag=True)
    
    return h_states
        
def load_mixed_dataset(split, *root):
    datasets = [AudioVisualDataset(None, None, cache_dir=cache_dir, split=split, transforms=T.Resize((32,32)),target_transforms=t_transforms) for cache_dir in root]
    combined = ConcatDataset(datasets)
    return combined

def test_sequence(dataloader, align_to='image'):
    total = 0
    correct = 0
    label_idx = 1 if align_to == 'audio' else 0
    #readout = a4_readout if align_to == 'audio' else it_readout
        
    with torch.no_grad():    
        for i, data in enumerate(dataloader, 0):
            x, label = data
            label = label[label_idx].to(device)
            #align_flag = torch.tensor([label_idx]*x[0].shape[0]).to(device)
            align_flag = torch.zeros(x[0].shape[0], 10, 16, 16).to(device) if label_idx == 0 else torch.ones(x[0].shape[0], 10, 16, 16).to(device)
            x.append(align_flag)
            x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]
            y = model(x)
            #y = model([x[1]])
            output = readout(y[0])

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    return correct/total

def train_sequence():
    running_loss = 0
        
    for i, data in enumerate(trainloader, 0):
        label_idx = torch.randint(0, 2, [1,])
        #readout = a4_readout if label_idx == 1 else it_readout
            
        optimizer.zero_grad()
            
        x, label = data
        label = label[label_idx].to(device)
        align_flag = torch.zeros(x[0].shape[0], 10, 16, 16).to(device) if label_idx == 0 else torch.ones(x[0].shape[0], 10, 16, 16).to(device)
        x.append(align_flag)
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
    parser.add_argument('--reciprocal', type = str2bool, default = True)
    parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/4_task_models/multimodal_brainlike.csv')
    parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

    parser.add_argument('--model_save', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/test.pt')
    parser.add_argument('--results_save', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/tes.npy')
    parser.add_argument('--hstates_save', type = str, default = None)

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
    
    uam_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uam'
    #uam_root = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_match'
    uan_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uan'
    #uan_root = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_mismatch'
    
    uum_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uum'
    #uun_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'
    uun_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uun'

    # %% [markdown]
    # # 1: Prepare dataset
    print('Loading datasets')
    
    # Load Audio
    #trainset = AudioVisualDataset(None, None, uun_root, split='train')
    trainset = load_mixed_dataset('train', aum_root, uam_root, uun_root)
    vs_testset = AudioVisualDataset(None, None, uam_root, split='test',transforms=T.Resize((32,32)))
    as_testset = AudioVisualDataset(None, None, aum_root, split='test',transforms=T.Resize((32,32)))
    mismatch_testset = AudioVisualDataset(None, None, uun_root, split='test',transforms=T.Resize((32,32)), target_transforms=t_transforms)
    control_testset = AudioVisualDataset(None, None, aun_root, split='test', transforms=T.Resize((32,32)), target_transforms=t_transforms)
    allamb_control = AudioVisualDataset(None, None, aan_root, split='test', transforms=T.Resize((32,32)), target_transforms=t_transforms)
    
    trainloader = DataLoader(trainset, batch_size=args['batch_sz'], shuffle=True)
    vs_testloader = DataLoader(vs_testset, batch_size=args['batch_sz'], shuffle=True)
    as_testloader = DataLoader(as_testset, batch_size=args['batch_sz'], shuffle=True)
    mismatch_testloader = DataLoader(mismatch_testset, batch_size=args['batch_sz'], shuffle=True)
    control_testloader = DataLoader(control_testset, batch_size=args['batch_sz'], shuffle=True)
    allamb_control_testloader = DataLoader(allamb_control, batch_size=args['batch_sz'], shuffle=True)

    # INIT GRAPH
    input_nodes = [0, 4, 7] # V1, A1
    output_node = [7]
    graph_loc = args['graph_loc']
    graph = Graph(graph_loc, input_nodes=input_nodes, output_nodes=output_node, reciprocal=args['reciprocal'])
    input_sizes = graph.find_input_sizes()
    input_dims = graph.find_input_dims()

    # INIT MODEL
    model = Architecture(graph, input_sizes, input_dims).cuda().float()
    readout = ClassifierReadout(model.output_sizes[0], n_classes=10).cuda().float()
    #a4_readout = ClassifierReadout(model.output_sizes[1], n_classes=10).cuda().float()

    params = [{'params': model.parameters(), 'lr': 0.001},
              {'params': readout.parameters(), 'lr': 0.001}]

    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()
    print(model)

    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in model_parameters])
    #print(params)

    losses = {'loss': [], 'train_acc': [], 'vs1_acc': [], 'vs2_acc': [], 'as1_acc': [], 'as2_acc': [], 
              'control_img_align': [], 'control_audio_align': []}    

    if os.path.exists(args['model_save']):
        model.load_state_dict(torch.load(args['model_save']))
        print("Loading existing ConvGRU model")
    else:
        print("No pretrained model found. Training new one.")

    for epoch in range(args['epochs']):
        #train_acc = test_sequence(trainloader, align_to=args['align_to'])
        train_acc = 0
        vs1_acc = test_sequence(vs_testloader, align_to='image')
        vs2_acc = test_sequence(mismatch_testloader, align_to='image')
        as1_acc = test_sequence(as_testloader, align_to='audio')
        as2_acc = test_sequence(mismatch_testloader, align_to='audio')
        control_img_align = test_sequence(allamb_control_testloader, align_to='image')
        control_audio_align = test_sequence(allamb_control_testloader, align_to='audio')
        #umap = dim_reduction(model, control_testset, datapoints=400)
        #with open('dim_red/audio/dim_red_tsne_MPC_control.npy', "wb") as f:
        #    pickle.dump(umap, f)
        #loss = train_sequence()
        loss = 0

        printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |VS1 accuracy {:1.5f} | VS2 accuracy {:1.5f} | AS1 accuracy {:1.5f} | AS2 accuracy {:1.5f} |control image alignment {:1.5f} |control audio alignment {:1.5f}'.format(epoch, loss, train_acc, vs1_acc, vs2_acc, as1_acc, as2_acc, control_img_align, control_audio_align)

        print(printlog)
        losses['loss'].append(loss)
        losses['train_acc'].append(train_acc)
        losses['vs1_acc'].append(vs1_acc)
        losses['vs2_acc'].append(vs2_acc)
        losses['as1_acc'].append(as1_acc)
        losses['as2_acc'].append(as2_acc)
        losses['control_img_align'].append(control_img_align)
        losses['control_audio_align'].append(control_audio_align)

        with open(args['results_save'], "wb") as f:
            pickle.dump(losses, f)

        torch.save(model.state_dict(), args['model_save'])
        
    if args['hstates_save'] != None:
        h_states = fetch_latents_all_tasks(model)
        
        with open(args['hstates_save'], "wb") as f:
            pickle.dump(h_states, f)

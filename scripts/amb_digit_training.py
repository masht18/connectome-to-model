import torch
import math
import pickle
import argparse
import random
import os
#import umap
from sklearn.manifold import TSNE

from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data_utils

from connectome_to_model.utils.datagen import *
from connectome_to_model.utils.mech_eval import fetch_hstates
from connectome_to_model.model.graph import Graph, Architecture
from connectome_to_model.model.readouts import ClassifierReadout
from connectome_to_model.utils.audio_dataset import MELDataset
from connectome_to_model.utils.audio_dataset import AudioVisualDataset

def fetch_latents_vs_tasks(model, areas=7, timesteps=7):
    h_states = torch.zeros(2, areas, timesteps)
    
    vs_testloader = DataLoader(match_testset, batch_size=400, shuffle=True)
    mismatch_testloader = DataLoader(mismatch_testset, batch_size=400, shuffle=True)
    
    h_states[0] = fetch_hstates(model, vs_testloader, align_to='image', areas=areas, timesteps=timesteps, attention_flag=False,)
    h_states[1] = fetch_hstates(model, mismatch_testloader, align_to='image', areas=areas, timesteps=timesteps, attention_flag=False,)
    
    return h_states


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def load_mixed_dataset(split, *root):
    datasets = [AudioVisualDataset(None, None, cache_dir=cache_dir, split=split) for cache_dir in root]
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

            output = model(x)
            output = readout(output[0])
            #print(label.shape)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
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
        output = readout(output[0])
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss, test_acc

def mech_eval(align_to='image'):
    eval_dataloader = DataLoader(match_testset, batch_size=1000, shuffle=False)
    reduced_hstates = torch.zeros(7, 7, 1000, 4)
    for t in range(7):
        with torch.no_grad():    
            x, label = next(iter(eval_dataloader))
            #label = torch.unsqueeze(label[0], 1)

            x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]

            output = model(x, process_time=t+1, return_all=True)
            output = [o.flatten(start_dim=1).cpu() for o in output]

            for area in range(7):
                #tsne = umap.UMAP()
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                if torch.count_nonzero(output[area]) == 0:
                    tsne_results = torch.zeros(1000, 2)
                else:
                    tsne_results = torch.tensor(tsne.fit_transform(output[area]))
                #reduced_hstates[t, area] = torch.cat((tsne_results, label), dim=1)
                reduced_hstates[t, area] = torch.cat([tsne_results,torch.unsqueeze(label[0], 1), torch.unsqueeze(label[1], 1)], dim=1)
                
    return reduced_hstates
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--layers', type = int, default = 1)
    parser.add_argument('--hidden_dim', type = int, default = 10)
    parser.add_argument('--reps', type = int, default = 1)
    parser.add_argument('--topdown', type = str2bool, default = True)
    parser.add_argument('--topdown_type', type = str, default = 'audio')
    parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/test/test_audio.csv')
    parser.add_argument('--connection_decay', type = str, default = 'ones')
    parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

    parser.add_argument('--model_save', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/test.pt')
    parser.add_argument('--results_save', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/test.npy')
    parser.add_argument('--readout_save', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/test.npy')
    parser.add_argument('--hstates_save', type = str, default = None)

    args = vars(parser.parse_args())

    # %%
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seed for reproducibility
    torch.manual_seed(args['seed'])
    torch.set_num_threads(1)
    print(device)

    # %% [markdown]
    # # 1: Prepare dataset
    print('Loading datasets')

    amb_match_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_match'
    clean_mismatch_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'
    amb_mismatch_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_mismatch'
    
    #amb_match_root='/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aum'
    #clean_mismatch_root='/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uun'
    #amb_mismatch_root='/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uan'

    trainset = load_mixed_dataset('train', amb_match_root, clean_mismatch_root)
    match_testset = AudioVisualDataset(None, None, cache_dir=amb_match_root, split='test')
    mismatch_testset = AudioVisualDataset(None, None, cache_dir=clean_mismatch_root, split='test')
    control_testset = AudioVisualDataset(None, None, cache_dir=amb_mismatch_root, split='test')
    
    print(len(trainset))

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    match_testloader = DataLoader(match_testset, batch_size=32, shuffle=False, num_workers=4)
    mismatch_testloader = DataLoader(mismatch_testset, batch_size=32, shuffle=False, num_workers=4) 
    control_testloader = DataLoader(control_testset, batch_size=32, shuffle=False, num_workers=4) 


    # INIT GRAPH
    input_nodes = [0, 4] # V1, A1
    output_node = 3 #IT
    graph_loc = args['graph_loc']
    graph = Graph(graph_loc, input_nodes=input_nodes, output_nodes=output_node, reciprocal=False)
    input_sizes = graph.find_input_sizes()
    input_dims = graph.find_input_dims()

    # INIT MODEL
    model = Architecture(graph, input_sizes, input_dims,
                        topdown=args['topdown']).cuda().float()
    readout = ClassifierReadout(model.output_sizes[0], n_classes=10).cuda().float()
    
    params = [{'params': model.parameters(), 'lr': 0.001},
              {'params': readout.parameters(), 'lr': 0.001}]
    
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()

    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in model_parameters])
    #print(params)

    losses = {'loss': [], 'train_acc': [], 'amb_match_acc': [], 'clean_mismatch_acc': [], 'ambimg_acc': [], 'ambimg_audio_align': []}    

    if os.path.exists(args['model_save']):
        model.load_state_dict(torch.load(args['model_save']))
        print("Loading existing ConvGRU model")
    else:
        print("No pretrained model found. Training new one.")
        
    if os.path.exists(args['readout_save']):
        readout.load_state_dict(torch.load(args['readout_save']))
        print("Loading existing readout")
    else:
        print("No pretrained readout found. Training new one.")

    for epoch in range(args['epochs']):
        #train_acc = test_sequence(trainloader)
        train_acc = 0
        match_acc = test_sequence(match_testloader)
        mismatch_acc = test_sequence(mismatch_testloader)
        ambimg_acc = test_sequence(control_testloader, align_to='image')
        ambimg_audio_alignment = test_sequence(control_testloader, align_to='audio')
        #umap = mech_eval()
        #with open('dim_red/dim_red_tsne_big_rnn_match.npy', "wb") as f:
        #    pickle.dump(umap, f)
        #loss, test_acc = train_sequence()
        loss = 0
        #test_acc = 0

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
        torch.save(readout.state_dict(), args['readout_save'])
        
    if args['hstates_save'] != None:
        h_states = fetch_latents_vs_tasks(model)
        
        with open(args['hstates_save'], "wb") as f:
            pickle.dump(h_states, f)

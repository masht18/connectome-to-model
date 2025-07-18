import os
import pickle
import torch
import random
import pandas as pd
import numpy as np
import argparse
#import seaborn as sns
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix

from connectome_to_model.utils.audio_dataset import AudioVisualDataset
from connectome_to_model.utils.mech_eval import neighborhood_hit_score
from connectome_to_model.model.graph import Graph, Architecture
from connectome_to_model.model.readouts import ClassifierReadout

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--graph_loc', type = str, default = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/graphs/ambaudio/multimodal_brainlike_MPC.csv')
    parser.add_argument('--save_hstates', type = str, default = 'dim_red/saved_hstates/dim_red_tsne_big_rnn_mismatch.npy')
    parser.add_argument('--reciprocal', type = str2bool, default = True)
    parser.add_argument('--align_to', type = str, default = 'image')

    parser.add_argument('--model_root', type = str, default = 'saved_models/4task_composite/')
    parser.add_argument('--model_type', type = str, default = 'brainlike_thickness')
    
    args = vars(parser.parse_args())

def fetch_hstates(model, eval_dataloader, 
                  align_to='audio', areas=8, timesteps=7,
                  device='cuda', attention_flag=False):
    h_states = torch.zeros(areas, timesteps)
    
    for t in range(timesteps):
        label_idx = 1 if align_to == 'audio' else 0
        
        with torch.no_grad():    
            x, label = next(iter(eval_dataloader))
            label = label[label_idx].to(device)
            
            if attention_flag:
                align_flag = torch.zeros(x[0].shape[0], 10, 16, 16).to(device) if label_idx == 0 else torch.ones(x[0].shape[0], 10, 16, 16).to(device)
                x.append(align_flag)

            x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]

            output = model(x, process_time=t+1, return_all=True)
            #if t == timesteps-1:
            #    preds = readout(output[3])
           # 
           #     _, predicted = torch.max(preds.data, 1)
           #     total = label.size(0)
           #     correct = (predicted == label).sum().item()

           #     print(total/correct)
            
            output = [o.flatten(start_dim=1).cpu() for o in output]
            
            for area in range(areas):
                nh = neighborhood_hit_score(output[area], label, k=6)
                #ss = silhouette_score(output[area].cpu(), label.cpu())
                #h_states[area, t] = torch.cat([nh, torch.tensor(ss)], dim=1)
                h_states[area, t] = nh
                
    
    return h_states
                
def run_all_tasks(model, areas=8, timesteps=7):
    h_states = torch.zeros(4, areas, timesteps)
    
    h_states[0] = fetch_hstates(model, vs_testloader, align_to='image', attention_flag=True)
    h_states[1] = fetch_hstates(model, as_testloader, align_to='audio', attention_flag=True)
    h_states[2] = fetch_hstates(model, mismatch_testloader, align_to='image', attention_flag=True)
    h_states[3] = fetch_hstates(model, mismatch_testloader, align_to='audio', attention_flag=True)
    
    return h_states

def run_tasks(model, areas=7, timesteps=7, align_to='image'):
    h_states = torch.zeros(2, areas, timesteps)
    s1_testloader = vs_testloader if align_to=='image' else as_testloader
    
    h_states[0] = fetch_hstates(model, s1_testloader, align_to=align_to)
    h_states[1] = fetch_hstates(model, mismatch_testloader, align_to=align_to)
    
    return h_states


aan_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aan'
aum_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aum'
aun_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aunv'
uam_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uam'
#uam_root = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_match'
uan_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uan'
uun_root = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uun'
#uun_root='/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'
t_transforms = lambda y: torch.tensor(y).to('cuda')

vs_testset = AudioVisualDataset(None, None, uam_root, split='test', transforms=T.Resize((32,32)))
as_testset = AudioVisualDataset(None, None, aum_root, split='test', transforms=T.Resize((32,32)))
mismatch_testset = AudioVisualDataset(None, None, uun_root, split='test', transforms=T.Resize((32,32)), target_transforms=t_transforms)

vs_testloader = DataLoader(vs_testset, batch_size=400, shuffle=True)
as_testloader = DataLoader(as_testset, batch_size=400, shuffle=True)
mismatch_testloader = DataLoader(mismatch_testset, batch_size=400, shuffle=True)

input_nodes = [0, 4] # V1, A1
output_node = [7]
graph= Graph(args['graph_loc'], input_nodes=input_nodes, output_nodes=output_node, reciprocal=args['reciprocal'])

input_sizes = graph.find_input_sizes()
input_dims = graph.find_input_dims()

# INIT MODEL
model = Architecture(graph, input_sizes, input_dims).cuda().float()
readout = ClassifierReadout(model.output_sizes[0], n_classes=10).cuda().float()

num_seeds = 4
num_tasks = 4
num_areas = 8
tsteps = 7

reduced_hstates = torch.zeros(num_seeds, num_tasks, num_areas, tsteps)
for i in range(1, num_seeds):
    model_type = args['model_type']
    model_path = os.path.join(args['model_root'], f'{model_type}_{i}.pt') 
    #readout_path = os.path.join(args['model_root'], f'{model_type}_{i}_classifier.pt')
    
    model.load_state_dict(torch.load(model_path))
    #readout.load_state_dict(torch.load(readout_path))
    reduced_hstates[i-1] = run_all_tasks(model, areas=num_areas, timesteps=tsteps)
    #reduced_hstates[i-1] = run_tasks(model, readout, areas=num_areas, timesteps=tsteps, align_to=args['align_to'])
    
with open(args['save_hstates'], "wb") as f:
    pickle.dump(reduced_hstates, f)

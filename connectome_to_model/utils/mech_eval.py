import umap
import torch
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

#from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix

def dim_reduction(model, testset, datapoints=1000, device='cuda'):
    eval_dataloader = DataLoader(testset, batch_size=datapoints, shuffle=False)
    reduced_hstates = torch.zeros(7, 7, datapoints, 4)
    for t in range(7):
        with torch.no_grad():    
            x, label = next(iter(eval_dataloader))
            #label = torch.unsqueeze(label[0], 1)

            x = [torch.unsqueeze(inp, 1).to(device).float() for inp in x if inp.ndim != 5]

            output = model(x, process_time=t+1, return_all=True)
            output = [o.flatten(start_dim=1).cpu() for o in output]

            for area in range(7):
                #tsne = umap.UMAP()
                tsne = TSNE(n_components=2, perplexity=20, random_state=42)
                if torch.count_nonzero(output[area]) == 0:
                    tsne_results = torch.zeros(datapoints, 2)
                else:
                    tsne_results = torch.tensor(tsne.fit_transform(output[area]))
                #reduced_hstates[t, area] = torch.cat((tsne_results, label), dim=1)
                reduced_hstates[t, area] = torch.cat([tsne_results,torch.unsqueeze(label[0], 1).cpu(), torch.unsqueeze(label[1], 1).cpu()], dim=1)
                
    return reduced_hstates
    
def neighborhood_hit_score(data, labels, k=4):
    # Calculate the distance matrix
    dist_matrix = distance_matrix(data, data)
    
    n = data.shape[0]
    hit_scores = torch.zeros(n)
    
    for i in range(n):
        # Sort the distances to find the k nearest neighbors (excluding the point itself)
        sorted_indices = np.argsort(dist_matrix[i])
        
        # Exclude the first point (itself)
        nearest_neighbors = sorted_indices[1:k+1]
        
        # Count how many of the k nearest neighbors share the same label
        #print(labels[nearest_neighbors] == labels[i])
        same_label_count = torch.count_nonzero(labels[nearest_neighbors] == labels[i])
        
        # Calculate the proportion of neighbors with the same label
        hit_scores[i] = same_label_count / k
    
    return torch.mean(hit_scores)


def fetch_hstates(model, eval_dataloader,
                  align_to='audio', 
                  datapoints=400, areas=8, timesteps=7,
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
            output = [o.flatten(start_dim=1).cpu() for o in output]
            
            for area in range(areas):
                nh = neighborhood_hit_score(output[area], label, k=6)
                #ss = silhouette_score(output[area].cpu(), label.cpu())
                #h_states[area, t] = torch.cat([nh, torch.tensor(ss)], dim=1)
                h_states[area, t] = nh
                
    return h_states


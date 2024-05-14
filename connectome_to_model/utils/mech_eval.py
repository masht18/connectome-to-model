import umap
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

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
    
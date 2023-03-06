'''
Testing models with
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
from model.topdown_gru import ConvGRUExplicitTopDown

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# data processing hyperparams
parser.add_argument('--frames_per_clip', type = int, default = 16)
parser.add_argument('--mel_sample_rate', type = int, default = 2)
parser.add_argument('--mel_channels', type = int, default = 64)
parser.add_argument('--mel_stft_hopsize', type = int, default = 64)
parser.add_argument('--mel_pad_length', type = int, default = 200)

parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--ucf_data', type = str, default = '/network/datasets/ucf101.var/ucf101_torchvision/UCF-101')
parser.add_argument('--ucf_annot', type = str, default = '/home/mila/m/mashbayar.tugsbayar/datasets/ucf101_annot')

parser.add_argument('--model_save', type = str, default = 'saved_models/audio_newdata.pt')
parser.add_argument('--results_save', type = str, default = 'results/no_topdown_newdata.npy')

args = vars(parser.parse_args())


# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

#################################################
# # 1: PREPARE DATASET
print('Loading datasets')

tfs = T.Compose([
            # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
            # scale in [0, 1] of type float
            T.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W) for easier convolutions
            T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            # rescale to the most common size
            T.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
])

#audio_transform = torchaudio.transforms.MelSpectrogram(args['mel_sample_size'], 
#                                     hop_length=args['mel_stft_hopsize'], 
#                                     n_fft=4*args['mel_channels'], 
#                                     n_mels=args['mel_channels'])

def custom_collate(batch):
    filtered_batch = []
    for video, audio, label in batch:
        #mel = torch.log(audio_transform(audio)+1e-6)/2.0
        
        # pad or trim
        #if mel.shape[-1]>=args['mel_pad_length']:
        #    mel = mel[:,:,:args['mel_pad_length']]
        #else:
        #    mel = pad_tensor(mel, args['mel_pad_length'], -1, pad_val=np.log(1e-6)/2.0)
            
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

train_data = datasets.UCF101(args['ucf_data'], args['ucf_annot'], 
                             frames_per_clip=args['frames_per_clip'],
                            train=True, transform=tfs)
test_data = datasets.UCF101(args['ucf_data'], args['ucf_annot'], 
                             frames_per_clip=args['frames_per_clip'],
                            train=False, transform=tfs)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


connection_strengths = [1, 1, 1, 1] 
criterion = nn.CrossEntropyLoss()

#connections = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
#node_params = [(1, 28, 28, 5, 5), (10, 15, 15, 5, 5), (10, 9, 9, 3, 3), (10, 3, 3, 3, 3)]

input_node = [0] # V1
output_node = 3 #IT
input_dims = [1, 0, 0, 0]
input_sizes = [(28, 28), (0, 0), (0, 0), (0, 0)]
graph_loc = '/home/mila/m/mashbayar.tugsbayar/convgru_feedback/sample_graph.csv'
graph = Graph(graph_loc, input_nodes=[0], output_node=3)
#graph = Graph(connections = connections, 
#              conn_strength = connection_strengths, 
#              input_node_indices = input_node, 
#             output_node_index = output_node,
#              input_node_params = node_params,
#              dtype = torch.cuda.FloatTensor)
model = Architecture(graph, input_sizes, input_dims).cuda().float()
optimizer = optim.Adam(model.parameters())

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

            video, label = data
            video, label = video.to(device), label.to(device)
                
            output = model([video])

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
            
        video, label = data
        video, label = video.to(device), label.to(device)

        # Generate random topdown for testing purposes only
        #topdown = torch.rand(imgs.shape[0], input_seqs.shape[1], args['topdown_c'], args['topdown_h'], args['topdown_w']).to(device)
        
        output = model([video])
            
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

np.loadtxt(os.path.join(data_dir,'HCP_labels.txt'),dtype=str)
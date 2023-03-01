import torch
import math
import pickle
import argparse
import random

from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data_utils
import os
from model.topdown_gru import ConvGRUExplicitTopDown

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
parser.add_argument('--epochs', type = int, default = 50)
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--task_prob', type = float, default = 0.5) # 1=fully ambiguous bottom-up, 0=clean bottom-up
parser.add_argument('--topdown_type', type = str, default = 'audio')
parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--model_save', type = str, default = 'saved_models/audio_newdata.pt')
parser.add_argument('--results_save', type = str, default = 'results/no_topdown_newdata.npy')

args = vars(parser.parse_args())
if args['topdown_type'] != 'text' and args['topdown_type'] != 'image' and args['topdown_type'] != 'audio':
    raise ValueError('Topdown style not implemented')

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

# %% [markdown]
# # 1: Prepare dataset
print('Loading datasets')

MNIST_path='../datasets/torchvision/'
transform = transforms.ToTensor()
clean_train_data = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=transform)
clean_test_data = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=transform)

# Label references, which help generate sequences
mnist_ref_train = generate_label_reference(clean_train_data)
mnist_ref_test = generate_label_reference(clean_test_data)

# Create dataloaders
amb_trainset = DatasetTriplet('/home/mila/m/mashbayar.tugsbayar/datasets/amnistV2', train=True)
amb_testset = DatasetTriplet('/home/mila/m/mashbayar.tugsbayar/datasets/amnistV2', train=False)
ambiguous_train = DataLoader(amb_trainset, batch_size=64, shuffle=True)
ambiguous_test = DataLoader(amb_testset, batch_size=64, shuffle=True)

clean_train_dataloader = DataLoader(clean_train_data, batch_size=64, shuffle=True)
clean_test_dataloader = DataLoader(clean_test_data, batch_size=64, shuffle=True)

# Subset clean data so it only has numbers used in ambiguous training
ambiguous_classes = [0, 1, 3, 4, 5, 6, 7, 8, 9]
test_indices = [indices for c in ambiguous_classes for indices in mnist_ref_test[c]]
clean_amb_class_test = Subset(clean_test_data, test_indices) 
clean_amb_class_test_dataloader = DataLoader(clean_amb_class_test, batch_size=64, shuffle=True)

train_indices = [indices for c in ambiguous_classes for indices in mnist_ref_train[c]]
clean_amb_class_train = Subset(clean_train_data, train_indices) 
clean_amb_class_train_dataloader = DataLoader(clean_amb_class_train, batch_size=64, shuffle=True)

# FSDD dataset in case of audio topdown
audio_ds = hub.load("hub://activeloop/spoken_mnist")
mel_ds = MELDataset(audio_ds)
audio_ref = generate_label_reference(audio_ds, dataset_type='fsdd')

print('Successfully loaded datasets')
# In case of text topdown 
text_labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
text_labels = str_to_bits(text_labels)

'''
Provide image or text clue, given labels
    
    :param reference_imgs (torchvision.Dataset)
        dataset of images to fetch clues from. If None, provide text clue
    :param dataset_ref (list)
        if providing image clue, provide label reference as well
        
'''
def fetch_clues(labels, clue_type='text', reference_imgs=None, dataset_ref=None):
    
    if clue_type == 'text':
        clues = torch.zeros(labels.shape[0], 27)
        for batch_idx in range(labels.shape[0]):
            clues[batch_idx] = text_labels[labels[batch_idx]]
    elif clue_type == 'image':   
        clues = torch.zeros((labels.shape[0], 1, 28, 28))
        for batch_idx in range(labels.shape[0]):
            clues[batch_idx] = reference_imgs[random.choice(dataset_ref[labels[batch_idx]]).item()][0]
    else:
        clues = torch.zeros((labels.shape[0], 1, 64, 64))
        for batch_idx in range(labels.shape[0]):
            clues[batch_idx] = mel_ds[random.choice(dataset_ref[labels[batch_idx]]).item()][0]
        
    return clues.to(device)

'''
Provide image or text clue, given labels
    
    :param dataloader
        dataloader to draw the target image from
    :param clean data (torchvision.Dataset)
        clean dataset to draw bottom-up sequence images from
    :param topdown_type ('image' or 'text' or 'audio')
    :param dataset_ref (list)
        if providing image clue, provide label reference as well
'''
>>>>>>> Stashed changes

train_data = data_utils.Subset(train_data, torch.arange(1024))
test_data = data_utils.Subset(test_data, torch.arange(512))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

def test_sequence(topdown_type = 'image', p_amb_vs_clean_first = 0.5, p_correct_topdown = 0.5):
    amb_iter = iter(ambiguous_test)
    clean_iter = iter(clean_amb_class_test_dataloader)
    running_loss = 0.0
        
    for i, data in enumerate(ambiguous_test, 0):
            imgs, label = data
            
            if label_style == 'ambiguous':
                pick = np.random.binomial(1, 0.5)
                imgs, label = imgs[1].to(device), label[:, pick].to(device)
            else:
                imgs, label = imgs.to(device), label.to(device)
                
            #input_seqs = sequence_gen(imgs, label, clean_data, mnist_ref_test, seq_style=seq_style)
            input_seqs = choice_sequence_gen(imgs, label, clean_test_data, mnist_ref_test, full_ambiguity=bottomup_ambiguity)
            
            if label_style == 'per-class':
                label = torch.argmax(label, 1)
                
            # Is the topdown going to relevant?
            if relevant_clue == True:
                clue_label = label
            else:
                clue_label = torch.randint(0, 10, (label.shape[0], ))
            
            # Generate the topdown signal based on given modality
            if topdown_type == 'audio':
                topdown = fetch_clues(clue_label, 'audio', audio_ds, audio_ref)
            elif topdown_type == 'image':
                topdown = fetch_clues(clue_label, 'image', clean_test_data, mnist_ref_test)
            else: 
                topdown = fetch_clues(clue_label, 'text')

            #print(input_seqs.shape)
            #print(topdown.shape)
            output = model(input_seqs.float(), topdown)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    
    return correct/total

def train_sequence(topdown_type = 'image', p_amb_vs_clean_first = 0.5, p_correct_topdown = 0.5):
    amb_iter = iter(ambiguous_train)
    clean_iter = iter(clean_amb_class_train_dataloader)
    running_loss = 0.0
        
    for i, data in enumerate(ambiguous_train, 0):
            
        optimizer.zero_grad()
            
        imgs, label = data
        pick = np.random.binomial(1, 0.5)
        imgs, label = imgs[1].to(device), label[:, pick].to(device)
            
        # Additive or random sequence?
        bottomup_ambiguity = np.random.binomial(1, p_amb_vs_clean_first)
        #if bottomup_clue == False:
        #    seq_style = 'addition'
        #else:
        #    seq_style = 'random'
            
        # Generate sequence
        #input_seqs = sequence_gen(imgs, label, clean_train_data, mnist_ref_train, seq_style=seq_style)
        input_seqs = choice_sequence_gen(imgs, label, clean_train_data, mnist_ref_train, full_ambiguity=bottomup_ambiguity)
            
        if bottomup_ambiguity  == True:  # If bottom-up was ambiguous, give correct topdown info based on label
            clue_label = label
        else:   # Else if bottom-up signal was correct, give the correct answer or a random answer
            #if np.random.binomial(1, p_correct_topdown) == 1:
            #    clue_label = label
            #else:
            clue_label = torch.randint(0, 10, (label.shape[0], ))

        # Generate the topdown signal based on given modality
        if topdown_type == 'audio':
            topdown = fetch_clues(clue_label, 'audio', audio_ds, audio_ref)
        elif topdown_type == 'image':
            topdown = fetch_clues(clue_label, 'image', clean_train_data, mnist_ref_train)
        else: 
            topdown = fetch_clues(clue_label, 'text')
            
        output = model(input_seqs.float(), topdown)
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss


if args['connection_decay'] == 'biological':
    connection_strengths = [15711/16000, 14833/16000, 9439/16000]
else:
    connection_strengths = [1, 1, 1]

losses = {'loss': [], 'val_topdown_only': [], 'val_add_only': [], 'val_add_topdown': [], 'val_none': []}    
criterion = nn.CrossEntropyLoss()
model = ConvGRUExplicitTopDown((28, 28), 10, input_dim=1, 
                               hidden_dim=10, 
                               kernel_size=(3,3),
                               connection_strengths=connection_strengths,
                               num_layers=2,
                               reps= 2, 
                               topdown=True, 
                               topdown_type='image',
                               dtype = torch.FloatTensor,
                               return_bottom_layer=True,
                               batch_first = False)
#model = ConvGRU((28, 28), 10, input_dim=1, hidden_dim=10, kernel_size=(3,3), num_layers=args['layers'], 
#                       dtype=torch.cuda.FloatTensor, batch_first=True).cuda().float()
optimizer = optim.Adam(model.parameters())


for epoch in range(1):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        x = x[None, :]
        
        #target = target[None, :]
        topdown = fetch_clues(torch.randint(0, 10, (target.shape[0], )))
        print("topdown size is",topdown.shape)
        out = model(x, topdown )

        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data * 0.1 #question: deleted [0] after data
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print ("==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx+1, ave_loss)")
                
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        x = x[None, :]
        out = model(x, fetch_clues(torch.randint(0, 10, (target.shape[0], ))))
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data * 0.1 #question: deleted [0] after data
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print ("==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx+1, ave_loss)")

torch.save(model.state_dict(), save_path)

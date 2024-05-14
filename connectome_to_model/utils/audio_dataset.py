'''
Dataset and helper functions for audiovisual task
'''

import os
import random

import torchaudio
import torch
import numpy as np
import torch.utils.data as tdata
from connectome_to_model.utils.datagen import generate_label_reference, label_reference_ambvisual

class AudioVisualDataset(tdata.Dataset):
    '''
    torch dataset with two streams of stimuli: audio and visual
    '''
    def __init__(self, visual_dataset, audio_dataset, 
                 cache_dir, split='train', match=True, audio_align=True,
                 cache=False, pad_length=32, device='cuda',
                 visual_ambiguity=False, audio_ambiguity=False,
                 transforms=None, target_transforms=None):
        
        self.visual = visual_dataset
        self.audio = audio_dataset
        
        self.visual_ambiguity = visual_ambiguity
        self.match = match
        self.cache = cache
        self.pad_length = pad_length
        self.device = device
        
        self.transforms = transforms
        self.target_transforms = target_transforms
        
        if cache:
            dataset_type = 'amb_fsdd' if audio_ambiguity else 'fsdd'
            self.audio_ref = generate_label_reference(self.audio, 10, dataset_type=dataset_type)
            self.data, self.targets = self.generate_dataset_audio_align() if audio_align else self.generate_dataset()
            torch.save(self.data, f'{cache_dir}/{split}_data.pt')
            torch.save(self.targets, f'{cache_dir}/{split}_targets.pt')
        else:
            self.data = torch.load(f'{cache_dir}/{split}_data.pt')
            self.targets = torch.load(f'{cache_dir}/{split}_targets.pt')
            
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transforms is not None:
            #x = (self.transforms(x[0]), self.transforms(x[1]))
            x = (self.transforms(x[0]), x[1])
        if self.target_transforms is not None:
            y = (self.target_transforms(y[0]), self.target_transforms(y[1]))
        
        return x, y
    
    def generate_dataset(self):
        '''
        Internal method, use when generating data for the first time
        '''
        data = []
        targets = []
        
        for index in range(len(self.visual)):
            # Get visual stimulus
            if self.visual_ambiguity: 
                (_, img, _), labels = self.visual[index] # if drawing from the ambiguous dataset
                gt = random.randint(0, 1)                         # randomly pick one of the labels
                img_label = labels[gt]
            else:
                gt = random.randint(0, 1)
                img, labels = self.visual[index]
                img = img[gt*2]
                img_label = labels[gt]

            # Get audio stimulus
            if self.match:
                audio_label = img_label
            else:
                audio_label = torch.tensor(random.randint(0, 9))
                
            mel = self.audio[random.choice(self.audio_ref[audio_label])][0]
            
            if mel.shape[-1]>=self.pad_length:
                mel = mel[:,:self.pad_length]
            else:
                mel = pad_tensor(mel, self.pad_length, -1, pad_val=np.log(1e-6)/2.0)
            
            # Save img + audio + target combo
            data.append((img, torch.unsqueeze(mel, 0)))
            targets.append((img_label, audio_label))

        return data, targets
    
    def generate_dataset_audio_align(self):
        '''
        Internal method, use when generating data for the first time
        '''
        visual_ref = label_reference_ambvisual(self.visual)
        data = []
        targets = []
        
        for index in range(len(self.audio)):
            
            # Get audio stimulus
            mel, audio_label = self.audio[index]
            img_label = audio_label if self.match else torch.tensor(random.randint(0, 9)).to(self.device)
            
            #sample from ambvisual triplet dataset
            partial_ds = visual_ref[img_label]
            img_idx = torch.randint(0, len(partial_ds), (1,))
            triplet, labels = partial_ds[img_idx] # if drawing from the ambiguous dataset

            if self.visual_ambiguity:
                img = triplet[1]
            else:
                label_idx = torch.where(labels.to(self.device) == img_label)[0]
                img = triplet[label_idx*2]
            
            #if mel.shape[-1]>=self.pad_length:
            #    mel = mel[:,:self.pad_length]
            #else:
            #    mel = pad_tensor(mel, self.pad_length, -1, pad_val=np.log(1e-6)/2.0)
            
            # Save img + audio + target combo
            data.append((img, torch.unsqueeze(mel, 0)))
            targets.append((img_label, audio_label))

        return data, targets

class AmbVisualDataset(tdata.Dataset):
    '''
    Streamlined wrapper for DatasetTriplet with one img (amb or clean) and one label
    '''
    def __init__(self, triplet_dataset, ambiguity=False):
        
        self.visual = triplet_dataset
        self.len = len(triplet_dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if ambiguity:
            (_, img, _), labels = self.visual[index]
            gt = random.randint(0, 1)                         # randomly pick one of the labels
            img_label = labels[gt]
        else:
            gt = random.randint(0, 1)
            img, labels = self.visual[index]
            img = img[gt*2]
            img_label = labels[gt]
        
        return img, img_label
    
'''
Modified from NFB's L’éclat du rire (The Sound of Laughter) code 
https://github.com/nfb-onf/sound-of-laughter
'''
class MELDataset(tdata.Dataset):

    def __init__(self, dataset, stft_hopsize=64, mel_channels=64, sample_rate=2,
                 transforms=None, pad_length=64, logmag=True, n_samples=None, device="cpu"):

        super(MELDataset, self).__init__()

        self.wav_db = dataset
        self.stft_hopsize = stft_hopsize
        self.mel_channels = mel_channels
        self.sample_rate = sample_rate
        self.n_fft = 4 * mel_channels
        self.n_samples = n_samples
        self.pad_length = pad_length
        self.device = device

        self.logmag = logmag

        # Todo: We can add data augmentation or cleaning techniques here
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate,
                                                            hop_length=stft_hopsize,
                                                            n_fft=self.n_fft,
                                                            n_mels=self.mel_channels)

        # Patch to mel filters to make it invertable with librosa
        #self.melspec.mel_scale.fb = torch.tensor(
        #    librosa.filters.mel(sample_rate, n_mels=self.mel_channels, n_fft=self.n_fft, norm=1).T
        #)

        self.transforms = transforms

        self.mels = {}

    #def mel2audio(self, mel):
    #    if self.logmag:
    #        mel = np.exp(2.0*mel)-1e-6
    #    return librosa.feature.inverse.mel_to_audio(mel, sr=self.sample_rate, n_fft=self.n_fft,hop_length=self.stft_hopsize, norm=1)

    def audio2mel(self, audio):
        mel = self.melspec(audio).detach()
        if self.logmag:
            mel = torch.log(mel+1e-6)/2.0
        return mel

    def __getitem__(self, idx):
        data = self.wav_db.audio[idx].data()
        label = self.wav_db.labels[idx].data()

        #if self.transforms is not None:
        #    data = self.transforms(data, self.sample_rate).astype(np.float32)
        data = torch.tensor(data, requires_grad=False).permute(1, 0)

        mel = self.audio2mel(data.float())

        # Truncate or pad
        if mel.shape[-1]>=self.pad_length:
            mel = mel[:,:,:self.pad_length]
        else:
            mel = pad_tensor(mel, self.pad_length, -1, pad_val=np.log(1e-6)/2.0)
        return mel.detach(), label

    def __len__(self):
        if self.n_samples:
            return min(self.n_samples, len(self.wav_db))
        return len(self.wav_db)

def pad_tensor(vec, pad, dim, pad_val=0.0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.ones(*pad_size)*pad_val], dim=dim)

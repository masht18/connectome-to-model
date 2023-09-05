import os
import soundfile
#import librosa
import gzip
import random

import torchaudio
import torch
import numpy as np
import torch.utils.data as tdata
from utils.datagen import generate_label_reference
from glob import glob

class AudioVisualDataset(tdata.Dataset):
    '''
    torch dataset that gives simultaneous image and audio clues
    
    '''
    def __init__(self, visual_dataset, audio_dataset, 
                 cache_dir, 
                 visual_ambiguity=False, audio_ambiguity=False, 
                 match=True, split='train', 
                 cache=False, 
                 stft_hopsize=64, mel_channels=64, sample_rate=2,
                 transforms=None, logmag=True, n_samples=None):
        
        self.visual = visual_dataset
        self.audio = audio_dataset
        
        self.visual_ambiguity = visual_ambiguity
        self.match = match
        
        # MELspec parameters
        self.stft_hopsize = stft_hopsize
        self.mel_channels = mel_channels
        self.sample_rate = sample_rate
        self.n_fft = 4 * mel_channels
        self.logmag = logmag
        self.pad_length = 32

        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate,
                                                            hop_length=stft_hopsize,
                                                            n_fft=self.n_fft,
                                                            n_mels=self.mel_channels)
        
        if cache:  
            self.audio_ref = generate_label_reference(self.audio, 10, dataset_type='fsdd')
            self.data, self.targets = self.generate_dataset()
            torch.save(self.data, f'{cache_dir}/{split}_data.pt')
            torch.save(self.targets, f'{cache_dir}/{split}_targets.pt')
        else:
            self.data = torch.load(f'{cache_dir}/{split}_data.pt')
            self.targets = torch.load(f'{cache_dir}/{split}_targets.pt')
            
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def audio2mel(self, audio):
        mel = self.melspec(audio).detach()
        if self.logmag:
            mel = torch.log(mel+1e-6)/2.0
        return mel
    
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

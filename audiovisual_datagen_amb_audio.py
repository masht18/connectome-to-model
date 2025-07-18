import torch
import hub

from ambiguous.dataset.dataset import DatasetTriplet

from torchvision import datasets
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import torchvision.transforms as T
from connectome_to_model.utils.audio_dataset import AudioVisualDataset, MELDataset

from amb_audio.datasets.datagen import AmbAudioDataset
from amb_audio.utils.transforms import TrimSilence, PadSpec, Log
from amb_audio.datasets.fsdd import TorchFSDDGenerator

from torchaudio.transforms import MelSpectrogram, MFCC
from torchvision.transforms import Compose

####################################
class WrapperDataset(Dataset):
    def __init__(self, data, targets):
        self.audio = data.data()
        self.targets = targets.data()
        print(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

####################################
# IMAGE
MNIST_path='/home/mila/m/mashbayar.tugsbayar/datasets'
#dataset = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=T.ToTensor())
#train_ds, val_ds = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
#test_img_ds = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=T.ToTensor())
transform = T.Resize((32, 32))
train_img_ds = DatasetTriplet('/network/scratch/m/mashbayar.tugsbayar/datasets/amnistV5', split='train', transform=transform)
test_img_ds = DatasetTriplet('/network/scratch/m/mashbayar.tugsbayar/datasets/amnistV5', split='test', transform=transform)

####################################
# CLEAN AUDIO
device = 'cuda'
playback_sample_rate = 8000
stft_hopsize = 64
mel_channels = 64
sample_rate = 2
n_fft = 4 * mel_channels
mel_params = {'sample_rate': sample_rate, 'mel_channels': mel_channels,
             'n_fft': n_fft, 'stft_hopsize': stft_hopsize}

melspec = MelSpectrogram(sample_rate, hop_length=stft_hopsize, n_fft=n_fft, n_mels=mel_channels)

# Create a transformation pipeline to apply to the recordings
transforms = Compose([
    TrimSilence(threshold=1e-6),
    melspec,
    Log(),
    PadSpec(pad_length=64)
    #MFCC(sample_rate=2, n_mfcc=32),
    #PadSpec(32)
])

# Fetch the latest version of FSDD and initialize a generator with those files
fsdd = TorchFSDDGenerator(version='local', transforms=transforms, load_all=True, path='recordings')

# Create two Torch datasets for a train-test split from the generator
train_mel_ds, _, test_mel_ds = fsdd.train_val_test_split(test_size=0.15, val_size=0.15)

####################################
# AMB AUDIO
amb_audio_trainset = AmbAudioDataset(None, None, '/network/scratch/m/mashbayar.tugsbayar/datasets/amb_audio', split='train', 
                                     cache=False, preload=True, dataset_sz=4000)
amb_audio_testset = AmbAudioDataset(None, None, '/network/scratch/m/mashbayar.tugsbayar/datasets/amb_audio', split='test', 
                                    cache=False, preload=True, dataset_sz=400)

###############################
# MULTIMODAL

# AMB/AMB MATCH
#cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aam'
#match=True
#visual_amb=True

#multimodal_ds_train = AudioVisualDataset(train_img_ds, amb_audio_trainset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                         match=match, split='train', audio_align=True, cache=True)
#multimodal_ds_test = AudioVisualDataset(test_img_ds, amb_audio_testset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                        match=match, split='test', audio_align=True, cache=True)


# AMB/AMB MISMATCH
#cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aan'
#match=False
#visual_amb=True

#multimodal_ds_train = AudioVisualDataset(train_img_ds, amb_audio_trainset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                         match=match, split='train', audio_align=True, cache=True)
#multimodal_ds_test = AudioVisualDataset(test_img_ds, amb_audio_testset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                        match=match, split='test', audio_align=True, cache=True)

# AMB AUDIO/UNAMB IMG MATCH

#cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aum'
#match=True
#visual_amb=False

#multimodal_ds_train = AudioVisualDataset(train_img_ds, amb_audio_trainset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                         match=match, split='train', audio_align=True, cache=True)
#multimodal_ds_test = AudioVisualDataset(test_img_ds, amb_audio_testset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                        match=match, split='test', audio_align=True, cache=True)

# AMB AUDIO/UNAMB IMG MISMATCH

#cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/aunv'
#match=False
#visual_amb=False

#multimodal_ds_train = AudioVisualDataset(train_img_ds, amb_audio_trainset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                         match=match, split='train', audio_align=True, cache=True)
#multimodal_ds_test = AudioVisualDataset(test_img_ds, amb_audio_testset, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=True,
#                                        match=match, split='test', audio_align=True, cache=True)

# AMB IMG/UNAMB AUDIO MATCH

cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uam'
match=True
visual_amb=True

multimodal_ds_train = AudioVisualDataset(train_img_ds, train_mel_ds, cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='train', audio_align=True, cache=True)
multimodal_ds_test = AudioVisualDataset(test_img_ds, test_mel_ds, cache_dir, visual_ambiguity=visual_amb,
                                        match=match, split='test', audio_align=True, cache=True)

# AMB IMG/UNAMB AUDIO MISMATCH

cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uan'
match=False
visual_amb=True

multimodal_ds_train = AudioVisualDataset(train_img_ds, train_mel_ds, cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='train', audio_align=True, cache=True)
multimodal_ds_test = AudioVisualDataset(test_img_ds, test_mel_ds, cache_dir, visual_ambiguity=visual_amb,
                                        match=match, split='test', audio_align=True, cache=True)

# UNAMB/UNAMB MISMATCH

#cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uun'
#match=False
#visual_amb=False

#multimodal_ds_train = AudioVisualDataset(train_img_ds, train_mel_ds, cache_dir, visual_ambiguity=visual_amb,
#                                         match=match, audio_align=True, split='train', cache=True)
#multimodal_ds_test = AudioVisualDataset(test_img_ds, test_mel_ds, cache_dir, visual_ambiguity=visual_amb,
#                                        match=match, audio_align=True, split='test', cache=True)

# UNAMB/UNAMB MATCH

#cache_dir = '/network/scratch/m/mashbayar.tugsbayar/datasets/audiovisual_brainlike/uum'
#match=True
#visual_amb=False

#multimodal_ds_train = AudioVisualDataset(train_img_ds, train_mel_ds, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=False,
#                                         match=match, audio_align=True, split='train', cache=True)
#multimodal_ds_test = AudioVisualDataset(test_img_ds, test_mel_ds, cache_dir, visual_ambiguity=visual_amb, audio_ambiguity=False,
#                                        match=match, audio_align=True, split='test', cache=True)
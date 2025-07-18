import torch
import hub

from ambiguous.dataset.dataset import DatasetTriplet

from torchvision import datasets
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import torchvision.transforms as T
from utils.audio_dataset import AudioVisualDataset, MELDataset
from utils.datagen import generate_label_reference

from torchfsdd import TorchFSDDGenerator, TrimSilence
from torchaudio.transforms import MFCC
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
dataset = DatasetTriplet('/network/scratch/m/mashbayar.tugsbayar/datasets/amnistV5', split='train', transform=transform)
train_ds, val_ds = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
test_ds = DatasetTriplet('/network/scratch/m/mashbayar.tugsbayar/datasets/amnistV5', split='test', transform=transform)
####################################
# AUDIO
# Create a transformation pipeline to apply to the recordings
transforms = Compose([
    TrimSilence(threshold=1e-6),
    MFCC(sample_rate=2, n_mfcc=32)
])

# Fetch the latest version of FSDD and initialize a generator with those files
fsdd = TorchFSDDGenerator(version='master', transforms=transforms, load_all=True)

# Create two Torch datasets for a train-test split from the generator
train_mel_ds, val_mel_ds, test_mel_ds = fsdd.train_val_test_split(test_size=0.15, val_size=0.15)

###############################
# MULTIMODAL
train_cache_dir = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_match'
test_cache_dir = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_match'

match=True
visual_amb=True

multimodal_ds_train = AudioVisualDataset(train_ds, train_mel_ds, train_cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='train', cache=True)
multimodal_ds_val = AudioVisualDataset(val_ds, val_mel_ds, train_cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='val', cache=True)
multimodal_ds_test = AudioVisualDataset(test_ds, test_mel_ds, test_cache_dir, visual_ambiguity=visual_amb,
                                        match=match, split='test', cache=True)

train_cache_dir = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_mismatch'
test_cache_dir = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_amb_mismatch'

match=False
visual_amb=True

multimodal_ds_train = AudioVisualDataset(train_ds, train_mel_ds, train_cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='train', cache=True)
multimodal_ds_val = AudioVisualDataset(val_ds, val_mel_ds, train_cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='val', cache=True)
multimodal_ds_test = AudioVisualDataset(test_ds, test_mel_ds, test_cache_dir, visual_ambiguity=visual_amb,
                                        match=match, split='test', cache=True)

train_cache_dir = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'
test_cache_dir = '/home/mila/m/mashbayar.tugsbayar/datasets/multimodal_clean_mismatch'

match=False
visual_amb=False

multimodal_ds_train = AudioVisualDataset(train_ds, train_mel_ds, train_cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='train', cache=True)
multimodal_ds_val = AudioVisualDataset(val_ds, val_mel_ds, train_cache_dir, visual_ambiguity=visual_amb,
                                         match=match, split='val', cache=True)
multimodal_ds_test = AudioVisualDataset(test_ds, test_mel_ds, test_cache_dir, visual_ambiguity=visual_amb,
                                        match=match, split='test', cache=True)
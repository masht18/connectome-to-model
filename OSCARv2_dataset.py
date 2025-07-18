#!/usr/bin/python
# _____________________________________________________________________________
#
#
# Copyright 2021 Markus Ernst
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#
# _____________________________________________________________________________


# standard libraries
# -----

import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils

from PIL import Image
import matplotlib.pyplot as plt


# custom functions
# -----
def show_batch(sample_batched):
	"""
	sample_batched: Tuple[torch.tensor, torch.tensor] -> None
	show_batch takes a contrastive sample sample_batched and plots
	an overview of the batch
	"""

	grid_border_size = 2
	nrow = 10

	batch_1 = sample_batched[0][0][:, 0:, :, :]
	batch_2 = sample_batched[0][1][:, 0:, :, :]
	difference = np.abs(batch_1 - batch_2)

	titles = ["left eye", "right eye", "difference"]

	fig, axes = plt.subplots(1, 3, figsize=(2 * 6.4, 4.8))
	fig.suptitle("Batchsize: 100")
    
	for (i, batch) in enumerate([batch_1, batch_2, difference]):
		ax = axes[i]
		grid = utils.make_grid(batch, nrow=nrow, padding=grid_border_size)
		ax.imshow(grid.numpy().transpose((1, 2, 0)))
		ax.set_title(titles[i])
		ax.axis("off")
        
	plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard Usecase
# -----
# For the standard usecase of having just one class you can use the built-in
# torchvision.datasets.ImageFolder dataset


def raw_reader(path):
	with open(path, 'rb') as f:
		bin_data = f.read()
	return bin_data

def pil_loader(path):
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')
		
class StereoImageFolder(Dataset):
	"""Modified ImageFolder Structure to Import Stereoscopic Data"""

	def __init__(self, root_dir, train, stereo=False, loader=pil_loader, transform=None, target_transform=None, nhot_targets=False):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""

		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		self.paths_to_left_samples = []
		self.paths_to_right_samples = []
		self.height = 32
		self.width = 32
		self.loader = loader
		self.stereo = stereo
		self.nhot = nhot_targets
		
		# move through the filestructure to get a list of all images
		self._add_data(root_dir)

	def _add_data(self, root_dir):
		root_dir = root_dir + '/train/left/' if self.train else root_dir + '/test/left/'
		objectclasses = os.listdir(root_dir)
		new_left_samples = []
		try:
			objectclasses.remove('.DS_Store')
		except(ValueError):
			pass
		for cla in objectclasses:
			class_folder = os.path.join(root_dir, cla)

			filenames = os.listdir(class_folder)
			try:
				filenames.remove('.DS_Store')
			except(ValueError):
				pass
			for name in filenames:
				new_left_samples.append(os.path.join(root_dir, cla, name))
				self.paths_to_left_samples.append(
					os.path.join(root_dir, cla, name))
					
		
		for item in new_left_samples:
			self.paths_to_right_samples.append(item.split('left')[0] + 'right' + item.split('left')[1])
	
	def _remove_data(self, n_samples, last_samples=True):
		for i in range(n_samples):
			if last_samples:
				self.paths_to_left_samples.pop()
				self.paths_to_right_samples.pop()
			else:
				self.paths_to_left_samples.pop(0)
				self.paths_to_right_samples.pop(0)
	
	def __len__(self):
		return len(self.paths_to_left_samples)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = self.paths_to_left_samples[idx]
		image = self.loader(img_name)
		
		target = []

		if self.nhot:
			t_list = self.paths_to_left_samples[idx].rsplit('.',1)[0].rsplit('-',3)[-3:]
			if len(t_list)==1:
				raise NotImplementedError('nhot targets not implemented for this dataset')
			target = np.array(t_list, dtype=np.int64)
			
		else:
			t_list = self.paths_to_left_samples[idx].rsplit('_', 1)[-1].rsplit('/')[0]
			
			if t_list.__class__ == str:
				target = np.array(t_list, dtype=np.int64) # target.append(int(t_list))
			else:
				for t in self.paths_to_left_samples[idx].rsplit('_', 1)[-1].rsplit('/')[0]:
					target.append(int(t))
				target = np.array(target, dtype=np.int64)
		
		
		if self.target_transform is not None:
			target = self.target_transform(target)


		if self.stereo:
			image_l = image
			image_r = self.loader(self.paths_to_right_samples[idx])
			
			if self.transform is not None:
				image_l = self.transform(image_l)
				image_r = self.transform(image_r)
			
			sample = [(image_l, image_r), target]


		else:
			if self.transform is not None:
				image = self.transform(image)
			
			sample = [image, target]

		return sample




if __name__ == "__main__":
	# OSMNIST2 Dataset
	# -----

	dataset = StereoImageFolder(
		root_dir='./osmnist2r/',
		train=False,
		stereo=True,
		transform=transforms.ToTensor()
		)

	dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
	for ibatch, sample_batched in enumerate(dataloader):
		show_batch(sample_batched)
		if ibatch == 3:
			break

	pass
	
# _____________________________________________________________________________

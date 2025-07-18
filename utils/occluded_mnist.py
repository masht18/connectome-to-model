import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, transforms


def occlude(image, x, y, occlusion_size):
    # Find the bounding box of the digit
    non_zero_pixels = np.argwhere(image > 0)
    if non_zero_pixels.size == 0:
        # If the image is completely black, occlude the entire image
        return np.zeros_like(image)
    else:
        (top_row, left_col), (bottom_row, right_col) = non_zero_pixels.min(0), non_zero_pixels.max(0) + 1
    
    # Compute the center of the bounding box
    center_x = (top_row + bottom_row) // 2
    center_y = (left_col + right_col) // 2
    
    # Compute the offset from the center to the occlusion position
    x_offset = x - center_x
    y_offset = y - center_y
    
    # Create a mask for the occluded region
    mask = np.zeros_like(image)
    mask[center_x+x_offset:center_x+x_offset+occlusion_size, center_y+y_offset:center_y+y_offset+occlusion_size] = 1
    
    # Occlude the image with the mask
    occluded = np.copy(image)
    occluded[mask == 1] = 0
    return occluded


class OccludedMNIST(Dataset):
    def __init__(self, root, occlusion_size=10, num_occlusions=5, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.occlusion_size = occlusion_size
        self.num_occlusions = num_occlusions
        self.mnist_dataset = datasets.MNIST(root=self.root, train=self.train, transform=None)
        
    def __getitem__(self, index):
        image, target = self.mnist_dataset[index]
        for i in range(self.num_occlusions):
            x = np.random.randint(0, image.shape[0] - self.occlusion_size)
            y = np.random.randint(0, image.shape[1] - self.occlusion_size)
            image = occlude(image, x, y, self.occlusion_size)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
        
    def __len__(self):
        return len(self.mnist_dataset)
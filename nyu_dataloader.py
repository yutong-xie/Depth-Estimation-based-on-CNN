import h5py
import numpy as np
import random
import torch
import torch.utils.data as data
from torchvision import transforms, utils
from PIL import Image

def LoadData(mode):
    path = 'dataset/nyu_depth_v2_labeled.mat'
    dataset = h5py.File(path, 'r')

    images= dataset["images"]
    depths = dataset["depths"]

    images = np.array(images)
    depths = np.array(depths)

    train, val, test = 1000, 250, 200

    if mode == "train":
        image = images[:train, :]
        depth = depths[:train, :]
    elif mode == "val":
        image = images[train:train + val, :]
        depth = depths[train:train + val, :]
    elif mode == "test":
        image = images[train + val + 190:, :]
        depth = depths[train + val + 190:, :]

    print("Total %s samples: %d" % (mode, len(image)))
    print("Image sample size:", image[0].shape)

    return (image, depth)


class NYUDataset(data.Dataset):
    def __init__(self, dataset, transform = None):
        self.images = dataset[0]
        self.depths = dataset[1]
        self.transform = transform

    def __getitem__(self, index):
        # image size: 3, 640, 480 (3, W, H)
        # depths size: 648, 480
        image = self.images[index]
        depth = self.depths[index]

        # Transpose the image to H, W, 3
        image = np.transpose(image, (2, 1, 0))
        depth = np.transpose(depth, (1, 0))

        image = Image.fromarray(image)
        depth = Image.fromarray(depth)

        # Apply transform
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        # Resize depth image to output size
        depth = transforms.Resize((74, 55))(depth)

        return image, depth

    def __len__(self):

        return len(self.images)

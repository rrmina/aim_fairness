# The Extended Yale B dataset is not publicly available
# Please download and extract before using this dataloader script
#
# Dataset Link: http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
# 
#   Please download the cropped version
# 

import os
import urllib.request as ur
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATASET_PATH = "data/"
DATASET_NAME = "CroppedYale"
BATCH_SIZE = 1

class PGMDataset(Dataset):

    URL = "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"
    DATASET_FILE_NAME = "CroppedYale.zip"

    def __init__(self, dataset_path="data/CroppedYale", download=True, transform=None):
        self.dataset_path = dataset_path

        # Download Dataset
        if (download):
            self._download_one(PGMDataset.URL)


        self.image_paths = []
        self.label_map = {}

        count = 0
        # Get the list of image paths
        for folder in os.listdir(dataset_path):
            class_folder = os.path.join(dataset_path, folder)

            # Label - Integer map
            self.label_map[class_folder.split('\\')[1]] = count

            for image_name in os.listdir(class_folder):

                # Skip non-PGM files. Some files .info
                if (image_name[-4:] == ".pgm"):
                    self.image_paths.append(os.path.join(class_folder, image_name))

            count += 1

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])

        print(self.label_map)
        print(self.image_paths[index].split('\\')[1])
        label = self.label_map[ self.image_paths[index].split('\\')[1] ]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_paths)

    def _download_one(self, url):
        file_path = self.dataset_path

        print(file_path.split('/')[0] + '/')

        if not os.path.exists(file_path.split('/')[0] + '/'):
            os.makedirs( file_path.split('/')[0] + '/')

        if not os.path.exists(self.dataset_path + ".zip"):
            print( "Downloading ", PGMDataset.DATASET_FILE_NAME, " ...")
            file_download = ur.URLopener()
            file_download.retrieve( PGMDataset.URL, file_path.split('/')[0] + '/' )

def test():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = PGMDataset(DATASET_PATH + DATASET_NAME, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for images, labels in dataloader:
        curr_batch_size = images.shape[0]


        print(labels)
        print(curr_batch_size)
        print(images.shape)

        break

test()
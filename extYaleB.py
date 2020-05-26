# The Extended Yale B dataset is not publicly available
# Please download and extract before using this dataloader script
#
# Dataset Link: http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
# 
#   Please download the cropped version
# 

import os
import urllib.request as ur
import zipfile

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

DATASET_PATH = "data/"

class PGMDataset(Dataset):

    URL = "http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip"
    DATASET_FILE_NAME = "CroppedYale.zip"
    DATASET_FOLDER = "CroppedYale/"

    def __init__(self, dataset_path="data/", download=True, transform=None):
        self.dataset_path = dataset_path
        extended_dataset_path = dataset_path + PGMDataset.DATASET_FOLDER

        # Download Dataset
        if (download):
            self._download_one(PGMDataset.URL)

        # Extract Zip File
        if not os.path.exists(extended_dataset_path):
            self._extract_zip()

        self.image_paths = []
        self.label_map = {}

        count = 0
        # Get the list of image paths
        for folder in os.listdir(extended_dataset_path):
            class_folder = os.path.join(extended_dataset_path, folder)

            # Label - Integer map
            self.label_map[class_folder.split('/')[2]] = count

            for image_name in os.listdir(class_folder):

                # Skip non-PGM files. Some files .info
                if (image_name[-4:] != ".pgm"):
                    continue
                
                # Skip ambient files 
                if (image_name[-11:] != "Ambient.pgm"):
                    self.image_paths.append(os.path.join(class_folder, image_name))

            count += 1

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        label = self.label_map[ self.image_paths[index].split('/')[2].split("\\")[0] ]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_paths)

    def _download_one(self, url):
        file_path = self.dataset_path
        zip_path = self.dataset_path + PGMDataset.DATASET_FILE_NAME

        # "data/"
        if not os.path.exists(file_path):
            os.makedirs( file_path )

        # "data/CroppedYale.zip"    
        if not os.path.exists( zip_path ):
            print( "Downloading ", PGMDataset.DATASET_FILE_NAME, " ...")

            with ur.urlopen(PGMDataset.URL) as response, open(zip_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

    def _extract_zip(self):
        zip_path = self.dataset_path + PGMDataset.DATASET_FILE_NAME
        print("Extracting ", zip-path, " to ", self.dataset_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.dataset_path)

# Uncomment this to debug
# def test():
#     # Device
#     device = ("cuda" if torch.cuda.is_available() else "cpu")

#     # Dataset and Dataloader
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     dataset = PGMDataset(DATASET_PATH, download=True, transform=transform)
#     BATCH_SIZE = 1
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#     aye = []
#     for images, labels in dataloader:
#         # curr_batch_size = images.shape[0]
#         aye.append(str(images.shape))
        
#     print(list(set(aye)))

# test()
from AnnoyManager import BaseAnnoyManager
from HDF5DataManager import HDF5DataManager
import torch
from PIL import Image
import os

h5_file = 'info/features.h5'
root_directory = r'G:\idmm\img_resized_1M\cities_instagram'
fileManager = HDF5DataManager(h5_file,batch_size=1000)
print(fileManager.len())
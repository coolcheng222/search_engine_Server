from AnnoyManager import BaseAnnoyManager
from HDF5DataManager import HDF5DataManager
import torch
from PIL import Image
import os

h5_file = 'info/data_1000.h5'
feature_h5 = 'info/features.h5'
root_directory = r'D:\idmm\img_resized_1M\cities_instagram'
dataManager = HDF5DataManager(feature_h5,batch_size=1000)
fileManager = HDF5DataManager(h5_file,batch_size=1000)
res = dataManager.len()
# print(res)
annoyManager = BaseAnnoyManager(2048)
annoyManager.load_index('data/tree.ann')
inc,dis = annoyManager.find_nearest_neighbors(torch.tensor(dataManager.get(0)[1]),10)
# print(inc,dis)
files = fileManager.search(inc)
print(files)
def show_images(root_dir, relative_paths):
    for path in relative_paths:
        image_path = os.path.join(root_dir, path.decode())
        
        image = Image.open(image_path)
        image.show()
show_images(root_directory,files)
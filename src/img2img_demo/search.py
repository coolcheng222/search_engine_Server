import torch
import sys
sys.path.append('src/preprocessor/')
from HDF5DataManager import HDF5DataManager
from Image2Feature import BaseImage2Feature
from AnnoyManager import BaseAnnoyManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1000
dim = 2048
h5_file = f'info/data_{batch_size}.h5'
root_directory = r'D:\idmm\img_resized_1M\cities_instagram'
ann_file = 'data/tree.ann'

fileManager = HDF5DataManager(h5_file,batch_size=batch_size)
annoyManager = BaseAnnoyManager(dim)
annoyManager.load_index('data/tree.ann')
img2fea = BaseImage2Feature(device)

def search(image):
    feature = img2fea.process(image)
    feature = feature.cpu().detach().squeeze(0)
    inc,dis = annoyManager.find_nearest_neighbors(feature,10)
    # print(inc,dis)
    files = fileManager.search(inc)
    return files,dis


import search
import os
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import sys
sys.path.append('src/preprocessor')
from HDF5DataManager import HDF5DataManager
from Image2Feature import BaseImage2Feature
from AnnoyManager import BaseAnnoyManager

from search import Searcher

root_directory = r'F:/idmm/img_resized_1M/cities_instagram'
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_images(root_dir, relative_paths):
    fig, axes = plt.subplots(20, 5)
    fig.tight_layout()

    for i, path in enumerate(relative_paths):
        image_path = os.path.join(root_dir, path)
        image = Image.open(image_path)
        
        row = i // 5
        col = i % 5
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

    plt.show()
img = 'src/img2img_demo/2.jpg'
import cv2
import time
k = []
t = []
# config here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1000
dim = 2048
h5_file = f'info/data_{batch_size}.h5'
root_directory = r'D:\idmm\img_resized_1M\cities_instagram'
ann_file = 'data/tree.ann'

fileManager = HDF5DataManager(h5_file,batch_size=batch_size)
annoyManager = BaseAnnoyManager(dim)
annoyManager.load_index(ann_file)
img2fea = BaseImage2Feature(device)

searcher = Searcher(img2fea,fileManager,annoyManager)
searcher.init()
for i in tqdm(range(50,2001,50)):
    img = 'src/img2img_demo/2.jpg'
    img = cv2.imread(img)
    # print(img)
    # img = np.random.randint(0, 256, size=img.shape, dtype=np.uint`8)

    start = time.time()

    imgs,dis = searcher.search(img,i)
    # print(imgs)
    end = time.time()
    k.append(i)
    t.append(end - start)
plt.plot(k,t)
plt.show()
# show_images(root_directory,imgs)
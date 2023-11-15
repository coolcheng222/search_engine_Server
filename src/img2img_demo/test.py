from search import search
import os
from PIL import Image

root_directory = r'G:\idmm\img_resized_1M\cities_instagram'
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_images(root_dir, relative_paths):
    fig, axes = plt.subplots(2, 5)
    fig.tight_layout()

    for i, path in enumerate(relative_paths):
        image_path = os.path.join(root_dir, path)
        image = Image.open(image_path)
        
        row = i // 5
        col = i % 5
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

    plt.show()
img = 'src/img2img_demo/3.jpg'
import cv2
img = cv2.imread(img)
imgs,dis = search(img)
print(imgs,dis)
show_images(root_directory,imgs)
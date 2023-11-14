from ImageIterator import ImageIterator
from HDF5DataManager import HDF5DataManager
from tqdm import tqdm
from Image2Feature import BaseImage2Feature
from AnnoyManager import BaseAnnoyManager
import torch
root_directory = r'D:\idmm\img_resized_1M\cities_instagram'
batch_size = 1000
feature_batch_size = 100

img_h5_file = f'info/data_{batch_size}.h5'
feature_h5 = 'info/features.h5'
annoy_file = 'data/tree.ann'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img2fea = BaseImage2Feature(device)
annoyManager = BaseAnnoyManager(dim = 2048)


def batchData(from_start = True):
    # if from_start == False, the process may continue from checkpoint automatically
    if from_start:
        checkpoint_file = None
    else:
        checkpoint_file = "checkpoint.pkl"

    dataManager = HDF5DataManager(img_h5_file,batch_size=batch_size)
    imageIterator = ImageIterator(root_directory, batch_size, checkpoint_file)
    # here is the process that save path batches to the images
    for batch_images in imageIterator:
        if len(batch_images) == 0:
            continue
        dataManager.save_start(batch_images,from_start)
    dataManager.save_end()
    print("batch over: " + str(dataManager.len()))
def featureExtration(from_start = True):
    checkpoint_filename = "checkpoint2.pkl"
    if from_start:
        checkpoint_file = None
    else:
        checkpoint_file = checkpoint_filename
    dataManager = HDF5DataManager(feature_h5)
    imageIterator = ImageIterator(root_directory, feature_batch_size, checkpoint_file)
    if not from_start:
        count = imageIterator.current_folder_idx * 100000 + imageIterator.current_image_idx
        count = count // 100
        print(f"{imageIterator.current_image_idx},{imageIterator.current_folder_idx}")
    else:
        count = 0
    for batch_images in imageIterator:
        if len(batch_images) == 0:
            continue
        feature = img2fea.preprocess(root_directory,batch_images)
        feature = feature.cpu().detach()
        dataManager.save_start(feature,from_start)
        imageIterator.save_checkpoint(checkpoint_filename)
        print("feature extracting: " + str(count))
        count += 1
    dataManager.save_end()
    print("feature over: " + str(dataManager.len()))
def putIntoAnnoy():
    dataManager = HDF5DataManager(feature_h5)
    number = dataManager.len()
    for i in tqdm(range(number)):
        a = dataManager.get(i)[:]
        # print(type(a))
        annoyManager.add_vectors(a)
    annoyManager.build_index()
    annoyManager.save_index(annoy_file)
    print(annoyManager.len())
# batchData(from_start=True) # here is the process that save path batches to the images(info/data.h5)
# featureExtration(from_start=False)
putIntoAnnoy()
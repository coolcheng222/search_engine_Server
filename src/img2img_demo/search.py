
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import os
from tqdm import tqdm
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import datasketch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaseImageSearch():
    def __init__(self,preprocessor):
        self.preprocessor = preprocessor
    def search(self,img,features,count=10):
        """return a image name list"""
        feature = self.preprocessor.process(img)
        # print(features[0])
        # print(feature)
        res = F.cosine_similarity(feature,features)
        values,keys = torch.topk(res,k = count)
        print(keys)
        print(values)

class BasePreProcessor():
    """
        This class is used for preprocess the pics in db(img->feature vector)(shape:[n,1000])
        should elaborate this because n == 1,000,000
        model:pic->feature model,here is resnet50(remove last hidden layer),no need to figure out details
        transforms: transform pic to a specific style(to input the model),no need to figure out the details
    """
    def __init__(self) -> None:
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model = self.model.to(device)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    def preprocess(self,path):
        """ preprocss----
            input img path,output the set of feature
            this may not work well when 1 million pics input"""
        batch = []
        image_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for image_file in tqdm(image_files[3500:3600]):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            i_image = self.transforms(image)
            # plt.imshow(i_image.permute(1, 2, 0))  # 将通道维度调整到最后
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()
            batch.append(i_image)
        batch_img = torch.stack(batch)
        batch_img = batch_img.to(device)
        # print(batch_img.shape)

        # with torch.no_grad():
        features = self.model(batch_img)
        features = torch.flatten(features, start_dim=1) 
        return features

    def process(self,image):
        """
            process one image(input search image)
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        features = self.model(image)
        features = torch.flatten(features, start_dim=1) 
        return features
class BaseLoader():
    """
        the interface to save and load the features(basic implements)
    """
    def __init__(self,dir):
        self.dir = dir
    def save(self,features):
        with open(os.path.join(self.dir,"abc.pickle"),"wb") as f:
            pickle.dump(features,f)
    def load(self):
        with open(os.path.join(self.dir,"abc.pickle"),"rb") as f:
            return pickle.load(f)

prep = BasePreProcessor()
loader = BaseLoader("./save/")
searcher = BaseImageSearch(prep)
features = prep.preprocess("./gallery/")
loader.save(features)
# features = loader.load()
# print(features.shape)
features = features.to(device)
print(device)
print(features.shape)

image = cv2.imread("0.jpg")

searcher.search(image,features)


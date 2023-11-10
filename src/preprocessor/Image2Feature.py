import torchvision.models as models
import torch
import torchvision.transforms as transforms
import cv2
import os
from tqdm import tqdm
class BaseImage2Feature():
    def __init__(self,device):
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
        self.device = device
    def preprocess(self,root,paths):
        """ preprocss----
            input img path,output the set of feature
            this may not work well when 1 million pics input"""
        batch = []
        for image_file in paths:
            image = cv2.imread(os.path.join(root,image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            i_image = self.transforms(image)
            batch.append(i_image)
        batch_img = torch.stack(batch)
        batch_img = batch_img.to(self.device)
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
        image = image.to(self.device)
        features = self.model(image)
        features = torch.flatten(features, start_dim=1) 
        return features

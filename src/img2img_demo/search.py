import numpy as np
class Searcher:
    def __init__(self,img2fea,fileManager,annoyManager):
        self.img2fea = img2fea
        self.fileManager = fileManager
        self.annoyManager = annoyManager
    def init(self):
        self.search(np.random.randint(0, 256, size=(300,300,3), dtype=np.uint8))
    def search(self,image,k = 10):
        feature = self.img2fea.process(image)

        feature = feature.cpu().detach().squeeze(0)

        inc,dis = self.annoyManager.find_nearest_neighbors(feature,k)
    
        # print(inc,dis)
        files = self.fileManager.search(inc)

        return files,dis


import os
import pickle
class BaseFeatureLoader():
    """
        the interface to save and load the features(basic implements)
    """
    def __init__(self,dir,step):
        self.dir = dir
        self.step = step
    def save(self,features,index):
        with open(os.path.join(self.dir,f"{index}.pickle"),"wb") as f:
            pickle.dump(features,f)
    def load(self,index):
        with open(os.path.join(self.dir,f"{index}.pickle"),"rb") as f:
            return pickle.load(f)
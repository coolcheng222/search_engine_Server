import h5py
class HDF5DataManager():
    """This one is for:
        1. search img after get the indices
        2. save the batch for preprocess
    """
    def __init__(self,filepath,batch_size=0):
        self.path = filepath
        self.file = None
        self.batch_size = batch_size
    def _open(self,read = True,from_start = False):
        if not self.file:
            if not read:
                self.file = h5py.File(self.path,'w' if from_start else 'a')
            else:
                self.file = h5py.File(self.path,'r')
    def save_start(self,batch_images,from_start):
        self._open(read = False,from_start = from_start)
        dataset_name = 'batch_' + str(len(self.file))
        self.file.create_dataset(dataset_name, data=batch_images)
    def save_end(self):
        self.file.close()
        self.file = None
    def len(self):
        self._open()
        length = len(self.file)
        self.save_end()
        return length
    def search(self,indices):
        self._open()
        res = []
        for index in indices:
            # 获取对应索引的数据
            item = self.file['batch_' + str(index // self.batch_size)][index % self.batch_size]
            res.append(item)
        self.save_end()
        return res
    def get(self,i):
        self._open()
        item = self.file[f'batch_{i}']
        # self.save_end()
        return item

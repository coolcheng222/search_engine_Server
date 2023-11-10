from HDF5DataManager import HDF5DataManager
h5_file = 'info/data.h5'
feature_h5 = 'info/features.h5'
dataManager = HDF5DataManager(feature_h5)
res = dataManager.get(0).shape
print(res)
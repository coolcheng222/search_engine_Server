from flask import Flask, jsonify, request
app = Flask(__name__)


from preprocessor.HDF5DataManager import HDF5DataManager
from preprocessor.Image2Feature import BaseImage2Feature
from preprocessor.AnnoyManager import BaseAnnoyManager
from img2img_demo.search import Searcher
from util import toCv2
import torch
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
searcherearcher.init()
@app.route('/search', methods=['POST'])
def hello():
    try:
        file = request.files['image']
        
        image_data = file.read()
        if len(image_data) == 0:
            return jsonify({"code":1,"msg": "No image received"})
        image = toCv2(image_data)
        imgs,dis = searcher.search(image,100)
        return jsonify(imgs)
    except Exception as e:
        print(e)
        return jsonify({"code": 1,"msg": "error"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
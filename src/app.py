import io
import os
from base64 import encodebytes
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import *
app = Flask(__name__)
CORS(app, supports_credentials=True)


from preprocessor.HDF5DataManager import HDF5DataManager
from preprocessor.Image2Feature import BaseImage2Feature
from preprocessor.AnnoyManager import BaseAnnoyManager
from img2img_demo.search import Searcher
from util import toCv2
import torch
import json
# config here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1000
dim = 2048

with open("config.json","r") as f:
    config = json.load(f)
    h5_file = config["h5_file"]
    root_directory = config["root_directory"]
    ann_file = config["ann_file"]

fileManager = HDF5DataManager(h5_file,batch_size=batch_size)
annoyManager = BaseAnnoyManager(dim)
annoyManager.load_index(ann_file)
img2fea = BaseImage2Feature(device)

searcher = Searcher(img2fea,fileManager,annoyManager)
searcher.init()

# Encode image into bytes
def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') 
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') 
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') 
    return encoded_img

# TODO: elaborate the api
@app.route('/search', methods=['POST'])
def hello():

    try:
        file = request.files.get('image', '')
        size = request.form['size']
        print(f'Search file numbers: {size}')

        image_data = file.read()
        if len(image_data) == 0:
            return jsonify({"code":1,"msg": "No image received"})
        image = toCv2(image_data)
        imgs,dis = searcher.search(image,int(size))
        encoded_imges = []
        for image_path in imgs:
            temp_image_dict = {}
            temp_encode = "data:image/jpg;base64," + get_response_image(os.path.join(root_directory,image_path))
            temp_image_dict["largeURL"] = temp_encode
            temp_image_dict["thumbnailURL"] = temp_encode
            temp_image_dict["width"] = 900
            temp_image_dict["height"] = 900
            encoded_imges.append(temp_image_dict)
        return jsonify(encoded_imges)
    except Exception as e:
        print(e)
        return jsonify({"code": 1,"msg": "error"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
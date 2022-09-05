from re import S
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image
import yaml
from yaml import Loader
import matplotlib.pyplot as plt
import json
import os
from preprocessing import adaptive_threshold_gaussian, noise_removal

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = '/home/daitama/Desktop/VietOCR/vietocr/transformerocr.pth'
config['device'] = 'cpu'

# Define detector
detector = Predictor(config)


path_to_image_folder = '/home/daitama/Desktop/VietOCR/vietocr/folder_images/after_rotated_train/'
with open('data.json') as data_file:
    data = json.load(data_file)
    for image_dir, file in data.items():
        image_dir = image_dir+'.jpg'
        n_boxes = len(file)
        # Path to image
        path_to_image = os.path.join(path_to_image_folder, image_dir)
        # Preprocess the image
        img = noise_removal(adaptive_threshold_gaussian(path_to_image))

        # Loop through the boxes:
        for box in range(n_boxes):
        # Get points of the cropped picture
            right = file[box]['crop'][0][0]
            bottom = file[box]['crop'][0][1]
            left = file[box]['crop'][1][0]
            top = file[box]['crop'][1][1]

            # Cropped image
            img_cropped = img.crop((left, top, right, bottom))

            prediction = detector.predict(img_cropped)
            
            file[box]['text'] = str(prediction)

with open('prediction.json', 'w') as fp:
    json.dump(data, fp)


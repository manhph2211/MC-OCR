import os
from re import S
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image
import json
import matplotlib.pyplot as plt
from typing import List
import sys
import pickle
# sys.path
sys.path.append('/home/daitama/Desktop/VietOCR/MC-OCR/text_recognition/vietocr')

from preprocessing import adaptive_threshold_gaussian, noise_removal

# Load Config for model
config = Cfg.load_config_from_name('vgg_transformer')
# Load weights. REMEMBER TO RESET WEIGHT PATH
config['weights'] = '/home/daitama/Downloads/transformerocr.pth'
config['device'] = 'cpu'


def infer_one_image():
    # Define detector
    detector = Predictor(config)

    # Infer for 1 image

    # Read JSON file
    # JSON file MUST HAVE THE SAME STRUCTURE AS GIVEN BEFORE

    path_to_json_file = '/home/daitama/Desktop/VietOCR/MC-OCR/data/demo/text_detection/data.json'
    path_to_image = '/home/daitama/Desktop/VietOCR/MC-OCR/data/demo/text_detection/mcocr_val_145114anqqj.jpg'

    with open(path_to_json_file) as data_file:
        data = json.load(data_file)
        image_dir, file = list(data.items())[0]
        n_boxes = len(file)

        # Preprocess the image
        img = noise_removal(adaptive_threshold_gaussian(path_to_image))

        # Loop through the boxes:
        for box in range(n_boxes):
        # Get points of the cropped picture
            right = file[box]['crop'][0][0]
            bottom = file[box]['crop'][0][1]
            left = file[box]['crop'][1][0]
            top = file[box]['crop'][1][1]

            # Crop the considering image
            img_cropped = img.crop((left, top, right, bottom))

            # Infer
            prediction = detector.predict(img_cropped)
            
            # Match the result with the field in JSON file
            data['../../data/val_images_after_semantic/mcocr_val_145114anqqj.jpg'][box]['text'] = str(prediction)
    return data


demo = infer_one_image()

with open('demo.json', 'w') as fp:
    json.dump(demo, fp, indent=2)
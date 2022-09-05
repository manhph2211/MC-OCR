import os
from re import S
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from PIL import Image
import json
import matplotlib.pyplot as plt
from typing import List
import sys
# sys.path
sys.path.append('/home/daitama/Desktop/VietOCR/MC-OCR/text_recognition/vietocr')

from preprocessing import adaptive_threshold_gaussian, noise_removal
print('hi')

a = adaptive_threshold_gaussian('/home/daitama/Desktop/VietOCR/MC-OCR/text_recognition/vietocr/image/harley.png')
plt.imshow(a, cmap='gray')
plt.show()
# # Load Config for model
# config = Cfg.load_config_from_name('vgg_transformer')
# # Load weights. REMEMBER TO RESET WEIGHT PATH
# config['weights'] = '/home/daitama/Downloads/transformerocr.pth'
# config['device'] = 'cpu'


# def infer_one_image(path_to_image, path_to_json_file):
#     # Define detector
#     detector = Predictor(config)

#     # Infer for 1 image

#     # Read JSON file
#     # JSON file MUST HAVE THE SAME STRUCTURE AS GIVEN BEFORE
#     path_to_json_file = path_to_json_file

#     with open(path_to_json_file) as data_file:
#         data = json.load(data_file)
#         image_dir, file = data.items()
#         image_dir = image_dir + '.jpg'
#         n_boxes = len(file)

#         # Path to image
#         path_to_image = path_to_image

#         # Preprocess the image
#         img = noise_removal(adaptive_threshold_gaussian(path_to_image))

#         # Loop through the boxes:
#         for box in range(n_boxes):
#         # Get points of the cropped picture
#             right = file[box]['crop'][0][0]
#             bottom = file[box]['crop'][0][1]
#             left = file[box]['crop'][1][0]
#             top = file[box]['crop'][1][1]

#             # Crop the considering image
#             img_cropped = img.crop((left, top, right, bottom))

#             # Infer
#             prediction = detector.predict(img_cropped)
            
#             # Match the result with the field in JSON file
#             file[box]['text'] = str(prediction)


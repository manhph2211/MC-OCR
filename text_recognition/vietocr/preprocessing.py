from http.client import ImproperConnectionState
import imgaug
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

path = '/home/daitama/Desktop/VietOCR/MC-OCR/text_recognition/vietocr/image/harley.png'

"""
We preprocess images by binarize and remove noise from them
"""

# Binarization
def adaptive_threshold_gaussian(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    new_img = Image.fromarray(new_img)
    return new_img

# Noise removal
def noise_removal(image):
    # Reading image from folder where it is stored 
    # denoising of image saving it into dst image 
    img = np.asarray(image)
    dst = cv2.fastNlMeansDenoising(img, None, 20,7,21) 
    return Image.fromarray(dst)


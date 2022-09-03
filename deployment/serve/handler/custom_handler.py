from ts.torch_handler.base_handler import BaseHandler
import os
import torch
import logging
import numpy as np
import cv2
from collections import defaultdict
logger = logging.getLogger(__name__)
import base64

class OCRHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        pass

    def inference(self, data, *args, **kwargs):
        return super().inference(data, *args, **kwargs)

    def initialize(self, context):
        pass

    def preprocess(self, requests):
        pass

    def preprocess_one_image(self, req):
        pass

    def inference(self, data, *args, **kwargs):
        pass

    def postprocess(self, data):
        pass

    def build_cluster_images(self, images):
        pass

    def process_image(self):
        pass

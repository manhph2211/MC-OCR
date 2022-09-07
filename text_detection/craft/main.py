"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
import json
from collections import defaultdict
import cv2
import numpy as np
from text_detection.craft import craft_utils
from text_detection.craft import imgproc
from text_detection.craft import file_utils
from text_detection.craft.craft import CRAFT


from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, square_size=1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


#==================================================================================
# load net
net = CRAFT()     # initialize
net.load_state_dict(copyStateDict(torch.load("text_detection/craft/ckpts/craft_mlt_25k.pth", map_location='cpu')))
net.eval()


def saveResult(img_file, img, boxes, dirname, verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = os.path.join(dirname, filename + '.txt')
        res_img_file = os.path.join(dirname, filename + '.jpg')

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)
              
        # Save result image
        cv2.imwrite(res_img_file, img)


def detect(img_path):
    image = imgproc.loadImage(img_path)
    bboxes, polys, score_text = test_net(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, cuda=False, poly=False, refine_net=None)
    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(img_path))
    file_utils.saveResult(img_path, image[:,:,::-1], polys, dirname="data/demo/text_detection")

    tmp = defaultdict(list)
    if True:
        file_txt = os.path.join("data/demo/text_detection/"+img_path.replace(".jpg",".txt").split("/")[-1])
        
        # keys_image = "_".join(file_txt.split('_')[:3])
        path_file_txt = file_txt#os.path.join(path_file,file_txt)
        print(path_file_txt)
        with open(path_file_txt, 'r') as f:
            for line in f:
                data = list(map(int, line.split(',')))
                x = data[::2]
                y = data[1::2]
                pt1 = [max(x), max(y)]
                pt2 = [min(x), min(y)]
                tmp[img_path].append({'crop': [pt1, pt2], 'text': ""})
            
    tmp = dict(tmp)
    json.dump(tmp, open('data/demo/text_detection/data.json', 'w', encoding='utf8'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    detect("data/demo/bg_sub/mcocr_val_145114aszbc.jpg")


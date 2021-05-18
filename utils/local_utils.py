import pandas as pd
import numpy as np
from ast import literal_eval
import json
import os
import cv2
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import jaccard_score
import torch
from torch.nn.functional import softmax


def read_json(path):
    with open(path, "r") as File:
        data = json.load(File)
        return data


def export_coco_format_data(annotation_path='./data/annotations/instances_default.json',
                            image_folder='./data/downloads/cats/',
                            mask_folder='./data/mask/',
                            export_data_path='./data/data.json'):
    coco = COCO(annotation_path)
    anns_ids = coco.getAnnIds()
    anns = coco.loadAnns(anns_ids)

    img_info = coco.imgs
    # group all annotations have same image id
    groups = {}
    for ann in anns:
        image_id = ann['image_id']
        if image_id not in groups:
            groups[image_id] = [ann]
        else:
            groups[image_id].append(ann)

    # save mask in for each image
    image_label_pairs = {}
    for image_id, ann_list in groups.items():
        image_mask = None
        for ann in ann_list:
            if image_mask is None:
                image_mask = coco.annToMask(ann)
            else:
                image_mask += coco.annToMask(ann)
        print(image_mask.shape)
        image_mask[image_mask > 1] = 1
        filename = img_info[image_id]['file_name']
        name, _ = os.path.splitext(filename)
        mask_path = os.path.join(mask_folder, name + '.png')
        cv2.imwrite(mask_path, image_mask)

        image_path = os.path.join(image_folder, filename)
        if not os.path.exists(image_path):
            raise ValueError('not found', image_path)
        image_label_pairs[image_path] =  mask_path

    with open(export_data_path, 'w') as f:
        json.dump(image_label_pairs, f)


def split_data(all_data_json_path='./data/data.json', ratio=[0.8, 0.15, 0.05]):
    with open(all_data_json_path, 'r') as f:
        data = json.load(f)

    all_image_paths = []
    all_mask_paths = []
    for image_path, mask_path in data.items():
        all_image_paths.append(image_path)
        all_mask_paths.append(mask_path)

    normalized_ratio = [e / sum(ratio) for e in ratio]
    train_image_paths, vt_image_paths, \
    train_mask_paths, vt_mask_paths = train_test_split(all_image_paths, all_mask_paths,
                                                       test_size=1 - normalized_ratio[0])
    val_image_paths, test_image_paths, \
    val_mask_paths, test_mask_paths = train_test_split(vt_image_paths, vt_mask_paths,
                                                       test_size=normalized_ratio[-1] / (1 - normalized_ratio[0]))

    def save_data(image_paths, mask_paths, save_path):
        data_dict = {}
        for image_path, mask_path in zip(image_paths, mask_paths):
            data_dict[image_path] = mask_path

        with open(save_path, 'w') as f:
            json.dump(data_dict, f)

    save_data(train_image_paths, train_mask_paths, './data/train_data.json')
    save_data(val_image_paths, val_mask_paths, './data/val_data.json')
    save_data(test_image_paths, test_mask_paths, './data/test_data.json')





if __name__=='__main__':
    df = pd.read_csv('./data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df.csv')
    annos = df['anno_polygons'].head(5).to_list()
    for anno in annos:
        anno = literal_eval(anno)
        for object in anno:
            category_id = object['category_id']
            print(category_id)




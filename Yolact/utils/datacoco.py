import torch
import torch.utils.data as data
import cv2
from PIL import Image
import json
import numpy as np

from utils.augmentation import train_aug, val_aug


def train_collate(batch):
    imgs, targets, masks = [], [], []
    valid_batch = [aa for aa in batch if aa[0] is not None]

    lack_len = len(batch) - len(valid_batch)
    if lack_len > 0:
        for i in range(lack_len):
            valid_batch.append(valid_batch[i])
    for sample in valid_batch:
        imgs.append(torch.tensor(sample[0], dtype=torch.float32))
        targets.append(torch.tensor(sample[1], dtype=torch.float32))
        masks.append(torch.tensor(sample[2], dtype=torch.float32))

    return torch.stack(imgs, 0), targets, masks


def val_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    targets = torch.tensor(batch[0][1], dtype=torch.float32)
    masks = torch.tensor(batch[0][2], dtype=torch.float32)

    return imgs, targets, masks, batch[0][3], batch[0][4]


def detect_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    return imgs, batch[0][1], batch[0][2]


class Receipt_Detection(data.Dataset):
    def __init__(self, cfg, mode="train"):
        self.mode = mode
        self.cfg = cfg

        if mode == "train":
            with open(self.cfg.train_json, "r") as F:
                data_dict = json.load(F)
            self.image_path = list(data_dict.keys())
            self.seg_path = [value[0] for value in list(data_dict.values())]
            self.list_bbox = [[value[1]] for value in list(data_dict.values())]
            print('len of image_train_path  : ', len(self.image_path))
            print('len of seg_train_path    : ', len(self.seg_path))
            print('len of list_box_train    : ', len(self.list_bbox))

        elif self.mode == "detect":
            self.img_path = cfg.test_image_path

    def __getitem__(self, index):
        if self.mode == 'detect':
            image = cv2.imread(self.img_path)
            image_normed = val_aug(image, self.cfg.img_size)
            return image_normed, image, self.img_path.split('/')[-1]

        img_path = self.image_path[index]
        seg_path = self.seg_path[index]
        img = cv2.imread(img_path)
        maskgt = Image.open(seg_path)
        maskgt = np.asarray(maskgt)
        h, w, _ = img.shape
        boxgt = self.list_bbox[index]

        box_list, mask_list, label_list = [], [], []
        for i, b in enumerate(boxgt):
            boxt = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            labelt = self.cfg.label_id[b[4]] - 1  # class to class_id
            maskt = (maskgt == i + 1).astype(float)  # create mask per object
            box_list.append(boxt.astype(float))
            label_list.append(labelt)
            mask_list.append(maskt)

        if len(box_list) > 0:
            boxes = np.array(box_list)
            masks = np.stack(mask_list, axis=0)
            labels = np.array(label_list)
            if self.mode == 'train':
                img, masks, boxes, labels = train_aug(img, masks, boxes, labels, self.cfg.img_size)
                if img is None:
                    return None, None, None
                else:
                    boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                    return img, boxes, masks

        else:
            print(f'No valid object in image: {index}. Use a repeated image in this batch.')
            return None, None, None

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_path)
        else:
            return 1

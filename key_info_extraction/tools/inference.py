import os
import json
import torch
from torch_geometric.loader import DataLoader
import cv2

from key_info_extraction.models.phobert_sage import SageNet
from key_info_extraction.datasets import Receipt
from key_info_extraction.models.phobert_gcn import BERTxGCN
from key_info_extraction.utils import ID2LABEL


model = BERTxGCN()
model.load_state_dict(torch.load("../gcn_best_epoch2.pth"))


def inference(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    dataset = Receipt(json_file=json_path)
    data = dataset[0]
    logit = model(data)
    pred = torch.argmax(logit, dim=1)

    key = list(json_data.keys())[0]
    for i in range(len(json_data[key])):
        json_data[key][i]['label'] = ID2LABEL[pred[i]]


def visualize(image_folder, json_path, save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_name = list(data.keys())[0] + '.jpg'
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)
    for box in data[image_name.split('.')[0]]:
        image = cv2.rectangle(image, (box['crop'][1][0], box['crop'][1][1]),
                      (box['crop'][0][0], box['crop'][0][1]), (0, 255, 0), 1)
        image = cv2.putText(image, 'OpenCV', (box['crop'][1][0], box['crop'][1][1]), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(image_path, image)


if __name__ == '__main__':
    pass

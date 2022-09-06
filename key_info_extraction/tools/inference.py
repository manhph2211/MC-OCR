import os
import json
import torch
import cv2
from key_info_extraction.models.phobert_sage import SageNet
from key_info_extraction.datasets import Receipt
from key_info_extraction.models.phobert_gcn import BERTxGCN



model = SageNet(768)
model.load_state_dict(torch.load("best_epoch2.pth"))

def decode(i):
    LABEL2ID = {
        0: "SELLER"     ,
        1: "ADDRESS"    ,
        2: "TIMESTAMP"  ,
        3: "TOTAL_COST" ,
        4: "OTHER"
    }

    return LABEL2ID[i]


def get_key(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    dataset = Receipt(json_file=json_path)
    data = dataset[0]
    logit = model(data)
    pred = torch.argmax(logit, dim=1)

    key = list(json_data.keys())[0]
    for i in range(len(json_data[key])):
        json_data[key][i]['label'] = decode(int(pred[i]))

    with open(json_path, 'w', encoding='utf') as f:
        json.dump(json_data, f)


def visualize(image_folder, json_path, save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_name = list(data.keys())[0] + '.jpg'
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)
    for box in data[image_name.split('.')[0]]:
        if box['label'] == 'OTHER':
            continue
        image = cv2.rectangle(image, (box['crop'][1][0], box['crop'][1][1]),
                      (box['crop'][0][0], box['crop'][0][1]), (0, 255, 0), 1)
        image = cv2.putText(image, box['text'] + ':' + box['label'], (box['crop'][1][0], box['crop'][1][1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 0), 1, cv2.LINE_AA)
              
    cv2.imwrite(save_path, image)


if __name__ == '__main__':
    get_key('../tests/demo.json')
    visualize('../tests', '../tests/demo.json', '../tests/test.jpg')
    pass

import pandas as pd
from tqdm import tqdm
import json


ID2LABEL = {
    15: "SELLER",
    16: "ADDRESS",
    17: "TIMESTAMP",
    18: "TOTAL_COST",
    0: "OTHER"
}

LABEL2ID = {
    "SELLER":       0,
    "ADDRESS":      1,
    "TIMESTAMP":    2,
    "TOTAL_COST":   3,
    "OTHER":        4
}


def compute_iou(box1, box2):
    box1 = [box1[1][0], box1[1][1], box1[0][0], box1[0][1]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    # determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    bb2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def create_data_annotation(json_path='data/demo/text_recognition/data.json',
                           label_path='data/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df.csv',
                           save_path='data/demo/kie/data.json'):
    labels = pd.read_csv(label_path)
    with open(json_path, 'rb') as f:
        data = json.load(f)

    for rowid in tqdm(labels.index):
        image_name = labels['img_id'][rowid].split('.')[0] + '.jpg'
        if image_name not in data.keys():
            continue
        try:
            anno_labels = labels['anno_labels'][rowid].split('|||')
        except:
            anno_labels = []

        for key, ubox in enumerate(data[image_name]):
            for id, label in enumerate(anno_labels):
                box = json.loads(labels['anno_polygons'][rowid].replace('\'', '\"'))[id]['bbox']
                if compute_iou(ubox['crop'], box) > 0.1:
                    data[image_name][key]['label'] = label
            if data[image_name][key].get('label', -1) == -1:
                data[image_name][key]['label'] = 'OTHER'

    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(data, f)


if __name__ == '__main__':
    create_data_annotation()
    with open('data/demo/kie/data.json', 'r') as f:
        data = json.load(f)
    print(data)

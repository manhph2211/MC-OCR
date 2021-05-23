import json

import torch
from torchvision.transforms import Normalize, ToTensor
import cv2
import os
from PIL import Image
from tqdm import tqdm
from itertools import groupby
from pycocotools.coco import COCO


os.chdir('/content/drive/MyDrive/RIVF2021-MC-OCR')


def image_transform(size, image):
    image = cv2.resize(image, dsize=(size, size)).astype(float) / 255
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    image = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])(image)

    return image


def mask_transform(size, mask):
    mask[mask > 0] = 1
    mask = cv2.resize(mask, (size, size))
    mask = torch.from_numpy(mask).long()

    return mask


def create_coco_annotation(json_coco_path='data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/annotations/instances_default.json',
                           json_split_path='data/train.json',
                           json_coco_save_path='data/coco_annotations/train.json'):
    with open(json_coco_path, 'r') as f:
        coco_annotations = json.load(f)
    origin_images = coco_annotations['images']
    origin_annotations = coco_annotations['annotations']

    with open(json_split_path, 'r') as f:
        file_names = list(json.load(f).keys())
    file_names = [file_name.split('/')[-1] for file_name in file_names]

    categories = [{
        'id': 1,
        'name': 'Receipt',
        'supercategory': ''
    }]
    images = [origin_images[i] for i in range(len(origin_images)) if origin_images[i]['file_name'] in file_names]
    image_ids = [images[i]['id'] for i in range(len(images))]
    annotations = [origin_annotations[i] for i in range(len(origin_annotations))
                   if origin_annotations[i]['image_id'] in image_ids]

    coco_annotations = {
        'categories': categories,
        'images': images,
        'annotations': annotations
    }

    with open(json_coco_save_path, 'w') as f:
        json.dump(coco_annotations, f)

    print(f'Save {len(coco_annotations["images"])} images and {len(coco_annotations["annotations"])} annotations')


def create_mask_image(annotation_path='data/annotations/instances_train.json',
                      save_path='data/mask_train/'):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        print(f'Create {save_path}')

    coco = COCO(annotation_path)
    imgIds = coco.getImgIds(catIds=[1])

    for i in tqdm(range(len(imgIds))):
        img = coco.loadImgs(imgIds[i])[0]
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=[1], iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = coco.annToMask(anns[0])
        for j in range(1, len(anns)):
            mask += coco.annToMask(anns[j])

        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

        save_mask_path = os.path.join(save_path, img['file_name'].split('/')[-1])
        mask.save(save_mask_path)


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return rle


def mask_to_annotation(gt_annotation='data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/annotations/instances_default.json',
                       mask_folder='data/mask',
                       save_path='data/mask.json'):
    with open(gt_annotation, 'r') as f:
        gt_annotations = json.load(f)

    gt_images = gt_annotations['images']
    gt_images = {gt_images[i]['file_name']: gt_images[i]['id'] for i in range(len(gt_images))}
    gt_annotations = gt_annotations['annotations']
    gt_annotations = {gt_annotations[i]['image_id']: gt_annotations[i]['id'] for i in range(len(gt_annotations))}

    images = list()
    annotations = list()
    categories = [{
        'id': 1,
        'name': 'Receipt',
        'supercategory': ''
    }]

    print('Create annotations for mask: ')
    for image_name, id in tqdm(gt_images.items()):
        if image_name not in os.listdir(mask_folder):
            continue
        image = dict()
        annotation = dict()

        image['id'] = id
        image_array = cv2.imread(os.path.join(mask_folder, image_name), cv2.IMREAD_GRAYSCALE)
        image_array[image_array > 0] = 1
        image['width'] = image_array.shape[1]
        image['height'] = image_array.shape[0]
        image['file_name'] = image_name

        annotation['id'] = gt_annotations[id]
        annotation['image_id'] = id
        annotation['category_id'] = 1
        annotation['segmentation'] = binary_mask_to_rle(image_array)

        contour, _ = cv2.findContours(image_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        box = cv2.boundingRect(contour[0])
        annotation['bbox'] = box
        annotation['area'] = int(image_array.sum())
        annotation['iscrowd'] = 0
        annotation['score'] = 1

        images.append(image)
        annotations.append(annotation)

    coco_annotations = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(save_path, 'w') as f:
        json.dump(coco_annotations, f)


if __name__ == '__main__':
    # image_paths = list(os.listdir('data/mask'))
    # mask = cv2.imread(os.path.join('data/mask', image_paths[2]), cv2.IMREAD_GRAYSCALE)
    # mask[mask > 0] = 1
    # mask = np.asfortranarray(mask)
    # mask = M.encode(mask)
    create_coco_annotation(json_coco_path='data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data'
                                          '/mcocr_train_data/annotations/instances_default.json',
                           json_split_path='data/train.json',
                           json_coco_save_path='data/coco_annotations/train.json')

    pass

from utils import mask_to_annotation
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from predict import inference_folder
import json
import os


os.chdir('/content/drive/MyDrive/RIVF2021-MC-OCR')


def evaluate(gt_annotation,
             checkpoint,
             mask_folder,
             save_annotation_path):
    with open(gt_annotation, 'r') as f:
        coco_annotations = json.load(f)
    images = coco_annotations['images']
    image_names = [images[i]['file_name'] for i in range(len(images))]

    inference_folder(image_names, checkpoint, mask_folder)
    mask_to_annotation(gt_annotation, mask_folder, save_annotation_path)

    gt_annotation = COCO(gt_annotation)
    dt_annotation = COCO(save_annotation_path)
    coco_eval = COCOeval(cocoGt=gt_annotation, cocoDt=dt_annotation)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
  evaluate(gt_annotation='data/coco_annotations/val.json',
           checkpoint='weights/checkpoint18.pth',
           mask_folder='mask_predict',
           save_annotation_path='mask_predict/mask.json')

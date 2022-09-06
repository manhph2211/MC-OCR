from model import get_instance_segmentation_model
import config
import glob
import os
from local_utils import remove_background
import cv2
from tqdm import tqdm 
import torch 
import math
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import sys


def rotate_image(image_path, boxes_path):
    image = cv2.imread(image_path)

    with open(boxes_path, 'r') as f:
        boxes = f.readlines()
    box_list = [box[:-1].split(',') for box in boxes if box != "\n"]
    try:
        box_list = np.array(box_list, dtype=int)
        bigger_x_axis = (box_list[:, 2] - box_list[:, 0]) > (box_list[:, 5] - box_list[:, 3])
        bigger_x_axis = sum(bigger_x_axis) > box_list.shape[0] // 2
    except IndexError:
        print(f'Ignore image path: {image_path}')
        return image

    if bigger_x_axis:
        index = np.argmax(box_list[:, 2] - box_list[:, 0])
        angle = math.atan(
            (box_list[index][3] - box_list[index][1]) / (box_list[index][2] - box_list[index][0])) * 180 / math.pi
    else:
        index = np.argmax(box_list[:, 5] - box_list[:, 3])
        angle = math.atan(
            (box_list[index][5] - box_list[index][3]) / (box_list[index][4] - box_list[index][2])) * 180 / math.pi

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)

    rotated_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotated_matrix, (width, height))

    return rotated_image


def convert(image_path, save_folder = "../../data/demo/text_detection"):
    name = image_path.split('/')[-1]
    box_path = os.path.join(save_folder, name.replace("jpg","txt"))
    return box_path


def remove_background_one_image(img_path, save_folder="../../data/demo/text_detection"):
  if type(img_path) == str:
    img = cv2.imread(img_path)
  # else:
    img = torch.tensor(img)
    img = img.permute(2,0,1) / 255
    name = img_path.split('/')[-1]
    with torch.no_grad():
      box_path = convert(img_path)
      print(box_path)
      if not os.path.isfile(box_path):
        prediction = model([img.to(config.device)])
        img = img.permute(1,2,0)
        prediction = prediction[-1]['masks'][0][0].cpu()
        remove_bg_img = remove_background(img,prediction) * 255
        cv2.imwrite(os.path.join('../../data/demo/bg_sub',name),remove_bg_img)
        print("DONE REMOVING BACKGROUND!!!")
      else:
        rotated_image = rotate_image(img_path, box_path)
        cv2.imwrite(os.path.join('../../data/demo/rotation',name),rotated_image)
        print("DONE ROTATION!!!")
    return os.path.join(save_folder,name)


def remove_background_dataset(original_folder,save_folder):
  img_paths = glob.glob(os.path.join(original_folder,'*.jpg'))
  for img_path in tqdm(img_paths):
      _ = remove_background_one_image(img_path,save_folder)


model = get_instance_segmentation_model(num_classes=config.n_classes)
model.to(config.device)
model.load_state_dict(torch.load(config.model_save_path, map_location = config.device))
model.eval()

# remove_background_dataset(config.train_imgs,config.save_train_img)
# remove_background_dataset(config.val_imgs,config.save_val_img)
if __name__ == "__main__":
  # remove_background_one_image("../../data/mcocr_public_train_test_shared_data/mcocr_val_data/val_images/mcocr_val_145114aszbc.jpg")
  remove_background_one_image("../../data/demo/bg_sub/mcocr_val_145114aszbc.jpg")

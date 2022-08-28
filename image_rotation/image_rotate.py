import math
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm


def rotate_image(image_path, boxes_path):
    image = cv2.imread(image_path)

    with open(boxes_path, 'r') as f:
        boxes = f.readlines()
    box_list = [box[:-1].split(',') for box in boxes]
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


def convert(image_path):
    box_path = image_path.split('/')
    box_path[-2] = 'text_detection_results'
    box_path[-1] = 'res_' + box_path[-1][:-4] + '.txt' 
    box_path = '/'.join(box_path)

    return box_path


if __name__ == "__main__":
    save_path = '../data/after_rotated_train'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for image_path in tqdm(glob.glob('../data/text_detection_results/*.jpg')):
        box_path = convert(image_path)

        rotated_image = rotate_image(image_path, box_path)
        new_image_path = image_path.replace('text_detection_results', 'after_rotated_train')
        cv2.imwrite(new_image_path, rotated_image)

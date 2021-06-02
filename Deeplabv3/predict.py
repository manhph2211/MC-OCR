from utils import image_transform
from model.deeplabv3 import DeeplabV3
import torch
import cv2
from tqdm import tqdm
import os


def inference(image_path, checkpoint):
    image = cv2.imread(image_path)
    original_size = image.shape
    image = image_transform(size=316, image=image)

    model = DeeplabV3(pretrained_backbone=False)
    model.eval()
    model.load_state_dict(torch.load(checkpoint))
    predict = model(image.unsqueeze(0))[0]

    predict = torch.argmax(predict, dim=0)
    predict = predict.numpy().astype(float)
    predict = cv2.resize(predict, (original_size[1], original_size[0]))

    return predict


def inference_folder(image_names, checkpoint, save_folder,
                     image_folder='data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images'):

    print(f'Inference process: ')
    model = DeeplabV3(pretrained_backbone=False)
    model.eval()
    model.load_state_dict(torch.load(checkpoint))

    for image_name in tqdm(image_names):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        original_size = image.shape
        image = image_transform(size=316, image=image)
        mask = model(image.unsqueeze(0))[0]

        mask = torch.argmax(mask, dim=0)
        mask = mask.numpy().astype(float)
        mask = cv2.resize(mask, (original_size[1], original_size[0])) * 255

        save_path = os.path.join(save_folder, image_name)
        cv2.imwrite(save_path, mask)


if __name__ == '__main__':
    pass
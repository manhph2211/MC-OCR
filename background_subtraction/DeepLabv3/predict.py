import torch
import cv2
from tqdm import tqdm
import os
import argparse
from glob import glob


from utils import image_transform
from model.deeplabv3 import DeeplabV3


def inference(image_folder, checkpoint, save_folder):
    for image_path in glob(image_folder):
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
        predict[predict > 0] = 1

        image = cv2.imread(image_path).float()
        image[..., 0] *= image
        image[..., 1] *= image
        image[..., 2] *= image

        save_path = os.path.join(save_folder, image_path.split('/')[-1])
        cv2.imwrite(save_path, image)


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


def get_parser():
    parser = argparse.ArgumentParser(
        description='Prediction phase'
    )

    parser.add_argument(
        '--image_folder',
        type=str,
        default=None,
        help='path to the image folder (default: None)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='path to the checkpoint (default: None)'
    )

    parser.add_argument(
        '--save_folder',
        type=str,
        default=None,
        help='folder to save image (default: None)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    inference(args.image_folder,
              args.checkpoint,
              args.save_folder)

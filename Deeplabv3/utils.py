import torch
from torchvision.transforms import Normalize
import cv2


def image_transform(size, image):
    image = cv2.resize(image, dsize=(size, size)).astype(float) / 255
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    image = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])(image)

    return image


def mask_transform(size, mask):
    mask[mask > 0] = 1
    mask = cv2.resize(mask, (size, size))
    mask = torch.from_numpy(mask).long()

    return mask

import matplotlib.pyplot as plt
from model.deeplabv3 import DeeplabV3
from utils import image_transform
import torch
import cv2


def inference(image_path, checkpoint):
    image = cv2.imread(image_path)
    original_size = image.shape
    image = image_transform(size=256, image=image)

    model = DeeplabV3()
    model.load_state_dict(torch.load(checkpoint))
    predict = model(image.unsqueeze(0))[0]

    predict = torch.argmax(predict, dim=0)
    predict = predict.numpy().astype(float)
    predict = cv2.resize(predict, (original_size[1], original_size[0]))

    plt.imshow(predict)

    return predict


if __name__ == '__main__':
    pass

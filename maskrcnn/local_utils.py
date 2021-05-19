import config
from PIL import Image
import json
import matplotlib.pyplot as plt 
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import numpy as np 


def read_json(path):
	with open(path,'r') as f:
		data = json.load(f)
	return data


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def remove_background(img,mask):
	img = np.asarray(img)
	mask = np.asarray(mask)
	return (img.transpose(2,0,1) * mask * 255).transpose(1,2,0)


if __name__ == '__main__':	
	data = read_json(config.export_data_test_path)
	img_path,mask_path = next(iter(data.items()))
	img = Image.open(img_path)
	mask = Image.open(mask_path)
	print(np.asarray(mask).shape)
	print(np.asarray(img).shape)
	removed_bg_img = remove_background(img,mask)
	mask.putpalette([0,0,0,255,0,0])
	plt.subplot(1,3,1)
	plt.imshow(img)
	plt.subplot(1,3,2)
	plt.imshow(mask)
	plt.subplot(1,3,3)
	plt.imshow(removed_bg_img)
	plt.show()



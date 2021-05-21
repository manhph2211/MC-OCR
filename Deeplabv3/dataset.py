from torch.utils.data import Dataset
import json
import cv2
import os


from utils import image_transform, mask_transform


os.chdir('/home/doanphu/Documents/Code/Practice/RIVF2021-MC-OCR')


class ReceiptDataset(Dataset):
    def __init__(self, size, annotations='data/train.json'):
        super(ReceiptDataset, self).__init__()
        self.size = size
        with open(annotations, 'r') as f:
            self.image_dict = list(json.load(f).items())

    def __getitem__(self, item):
        image_path, mask_path = self.image_dict[item]

        image = cv2.imread(image_path)
        image = image_transform(self.size, image)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask_transform(self.size, mask)

        return image, mask

    def __len__(self):

        return len(self.image_dict)


if __name__ == '__main__':
    dataset = ReceiptDataset(size=256)

    print(dataset[0])
    pass

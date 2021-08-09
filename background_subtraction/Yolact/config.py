import os
import numpy as np
import torch
import glob


norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)


COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')


class set_config:
    def __init__(self):
        self.img_size = 550
        self.classes = ('receipt',)
        self.num_classes = len(list(self.classes)) + 1
        self.label_id = {(aa+1): (aa + 1) for aa in range(self.num_classes - 1)}
        self.coef_dim = 32
        self.num_anchors = 3
        self.lr = 0.00001
        self.mode = 'train'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg_name = "resnet50"
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.scales = [int(self.img_size / 550 * aa) for aa in (32, 64, 128, 256, 512)]
        self.aspect_ratios = [1, 1 / 2, 2]
        self.data_root = '/content/drive/MyDrive/RIVF2021/RIVF2021-MC-OCR/data/'
        self.train_bs = 8
        self.img_path = self.data_root + "mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images/"
        self.train_json = self.data_root + "train.json"
        self.val_json = self.data_root + "val.json"
        self.test_json = self.data_root + "test.json"
        self.pos_iou_thre = 0.6
        self.neg_iou_thre = 0.4
        self.masks_to_train = 100
        self.resume = True
        self.weight_path = "/content/drive/MyDrive/RIVF2021/yolact/weight/"
        self.batch_size = 8
        self.test_image_path = '/content/drive/MyDrive/RIVF2021/yolact/test_img.png'
        self.video = None
        self.top_k = 200
        self.no_crop = False
        self.hide_mask = False
        self.hide_bbox = False
        self.hide_score = False
        self.cutout = False
        self.real_time = False
        self.save_lincomb = False
        self.visual_thre = 0.2
        self.nms_score_thre = 0.05
        self.nms_iou_thre = 0.3
        self.max_detections = 100

        self.conf_alpha = 1
        self.bbox_alpha = 1.5
        self.mask_alpha = 6.125
        self.semantic_alpha = 1


def save_latest(net, cfg_name, step, path="/content/drive/MyDrive/RIVF2021/yolact/weight/"):
    weight = glob.glob(path + 'latest*')
    weight = [aa for aa in weight if cfg_name in aa]
    assert len(weight) <= 1, 'Error, multiple latest weight found.'
    if weight:
        os.remove(weight[0])

    print(f'\nSaving the latest model as \'latest_{cfg_name}_{step}.pth\'.\n')
    torch.save(net.state_dict(), f'{path}/latest_{cfg_name}_{step}.pth')

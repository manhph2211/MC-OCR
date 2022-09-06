from model import get_instance_segmentation_model
import config
import glob
import os
from local_utils import remove_background
import cv2
from tqdm import tqdm 
import torch 


def remove_background_one_image(img_path, save_folder):
  if type(img_path) == str:
    img = cv2.imread(img_path)
  else:
    img = torch.tensor(img)
    img = img.permute(2,0,1) /255
    name = img_path.split('\\')[-1]
    with torch.no_grad():
      prediction = model([img.to(config.device)])
      img = img.permute(1,2,0)
      prediction = prediction[-1]['masks'][0][0].cpu()
      remove_bg_img = remove_background(img,prediction) * 255
      cv2.imwrite(os.path.join(save_folder,name),remove_bg_img)

  return os.path.join(save_folder,name)


def remove_background_dataset(original_folder,save_folder):
  img_paths = glob.glob(os.path.join(original_folder,'*.jpg'))
  for img_path in tqdm(img_paths):
      _ = remove_background_one_image(img_path,save_folder)


model = get_instance_segmentation_model(num_classes=config.n_classes)
model.to(config.device)
model.load_state_dict(torch.load(config.model_save_path, map_location = config.device))
model.eval()

remove_background_dataset(config.train_imgs,config.save_train_img)
remove_background_dataset(config.val_imgs,config.save_val_img)


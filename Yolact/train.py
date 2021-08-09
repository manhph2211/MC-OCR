import os
import time
import glob
from PIL import Image

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from modules.yolact import *
from config import *
from utils.datacoco import *
from utils.output_utils import *


torch.autograd.set_detect_anomaly(True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config
cfg = set_config()

# Dataset & Dataloader
train_dataset = Receipt_Detection(cfg, 'train')
train_data_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, collate_fn=train_collate)

# Model
model = Yolact(cfg).to(device)

# Optimizer
optimizer =optim.SGD(model.parameters(), lr= cfg.lr, weight_decay=5e-4)

# Create folder to save weight
if not os.path.exists(cfg.weight_path):
  os.mkdir(cfg.weight_path)

# Continue training or not
cfg.resume = True

if cfg.resume:
  wtemp = glob.glob("/content/drive/MyDrive/RIVF2021/yolact/weight/latest*")
  model.load_weights(wtemp[0])
  start_step = int(wtemp[0].split('.pth')[0].split('_')[-1])+1
  print(f'\nResume training with \'{start_step}\'.\n')

else:
  model.init_weights()
  print(f'\nTraining from begining, weights initialized \n')
  start_step = 0 
  cfg.resume = True

# Train  
train_mode = True
time_last = time.time()
step = start_step


while train_mode:
  # Train 
  model.train()
  for i, (images,boxes,masks) in enumerate(train_data_loader):
    if torch.cuda.is_available():
      images = images.cuda().detach()
      boxes = [b.cuda().detach() for b in boxes]
      masks = [mask.cuda().detach() for mask in masks]

    loss_c, loss_b, loss_m, loss_s = model(images, boxes, masks)
    loss_total = loss_c + loss_b + loss_m + loss_s


    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    time_this = time.time()
    batch_time = time_this-time_last
    if i%30==0:
      print("Time for batch " , i , "/", len(train_data_loader) , "= " , batch_time , ", Loss  = " , loss_total.item() )
    time_last = time.time()

  save_latest(model, cfg_name = cfg.cfg_name , step = step)
  print("Epoch no ",step ,"  completed")
  print("----------------------------------------------------")
  step += 1

MaskRCNN for Removing Background of Original Image
=====

# MaskRCNN

# How to train ?

- Change the `device` to `cuda` in the config file if you have one! Then just run `python3 train.py` 

# Saving Image after Removing Background

- First downloading [model](https://drive.google.com/file/d/1wYjxfS3PWyX8gKyN5Yg33YAzUycZeBJB/view?usp=sharing) and put it into `./maskrcnn/`


- Following these steps:
```
mkdir ../data/train_images_after_semantic
mkdir ../data/val_images_after_semantic
python3 save_img.py
```


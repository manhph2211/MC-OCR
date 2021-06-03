# Deeplabv3 for receipt segmentation

## Introduction to deeplab family 
Deeplab is a semantic segmentation model designed and open-source by
Google. To mitigate some challenges in dense prediction, this authors proposed 
some solutions:
- Reduce resolution: Deeplab applied Astrous convolution to enlarge the field-of-view
  without decimating resolution. 
- Objects in different scale: ASPP is a parallel network which uses astrous convolutions with multiple rates
  and incorporates them together.
- Smooth a mask prediction: CRFs is a dense graph model which can maximize label agreement between similar pixels, and
  can integrate more elaborate terms that model contextual relationships between object classes.

## Requirements
```bash
$ pip install -r requirements.txt
```

## Training 
```bash
$ python train.py --help

usage: train.py [-h] [--image_scale IMAGE_SCALE] [--num_epochs NUM_EPOCHS]
                [--batch_size BATCH_SIZE] [--lr LR] [--criterion CRITERION]
                [--save_checkpoint SAVE_CHECKPOINT]
                [--load_checkpoint LOAD_CHECKPOINT]

Training phase

optional arguments:
  -h, --help            show this help message and exit
  --image_scale IMAGE_SCALE
                        size of image after scaling (default: 312)
  --num_epochs NUM_EPOCHS
                        number of epochs to train (default: 30)
  --batch_size BATCH_SIZE
                        number of images used in an iteration (default: 8)
  --lr LR               learning rate (default: 1e-4)
  --criterion CRITERION
                        loss function (default: ce) [options: mse=mean square
                        error, fl=focal loss]
  --save_checkpoint SAVE_CHECKPOINT
                        path to the directory for saving checkpoints (default:
                        weights)
  --load_checkpoint LOAD_CHECKPOINT
                        path to the checkpoint for loading (default: None)
```

## Inference
```bash
$ python predict.py --help

usage: predict.py [-h] [--image_path IMAGE_PATH] [--checkpoint CHECKPOINT]

Prediction phase

optional arguments:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH
                        path to the image (default: None)
  --checkpoint CHECKPOINT
                        path to the checkpoint (default: None)
```

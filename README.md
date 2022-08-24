# RIVF2021-MC-OCR
The text recognition of Vietnamese receipts. The competition task aims at extracting required fields in Vietnamese receipts captured by mobile devices :smile:

## Pipeline

![image](https://user-images.githubusercontent.com/61444616/186398778-8d0019a7-3f9c-484d-a2da-6032ff1c8457.png)

Following these steps:

- [x] Anno label(cvat)
- [x] Semantic(FPN, Mask RCNN, Yolact)
- [x] Crop Receipt from original Image 
- [x] Detect texts(DB,CRAFT)
- [x] Rotate angle - Alignment
- [x] Binary Labels up, down (cvat)
- [ ] Handling imbalanced dataset & Binary Classifier (EfficientNetv2)
- [ ] Text Recognition - VietOCR/ AttOCR
- [ ] Key Infomation Extraction - GraphSage

## Dataset 

### Dataset for Semantic Segmentation

- First, you need to download [data](https://drive.google.com/file/d/1Ma-vnGBXOMMVa1n4Oyd79mywAmx2MvCe/view?usp=sharing ), put it into ./data

- `cd ./data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data && mkdir annotations`

- Download [annotation file](https://drive.google.com/file/d/1NpV5h9ZfhfkV1c7SL1I6iAhSVHC596yM/view?usp=sharing) and put it into folder `annotations`

- Then following these steps :

```
cd data && mkdir mask
cd ../utils/
python3 local_utils.py

```


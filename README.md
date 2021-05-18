# RIVF2021-MC-OCR
The text recognition of Vietnamese receipts. The competition task aims at extracting required fields in Vietnamese receipts captured by mobile devices :smile:


## Pipeline
Following these steps:

- Anno label(cvat)
- Semantic(FPN, Mask RCNN, Yolact)
- Crop Receipt from original Image 
- Detect texts(DS,CRAFT)
- Rotate angle
- Binary Labels up, down (cvat)
- Handling imbalanced dataset
- Binary Classifier (RepVGG,EfficientNetv2)
- VietOCR
- Key Extraction



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


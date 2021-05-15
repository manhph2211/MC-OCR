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

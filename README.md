MC-OCR
====

# Introduction

In this project, we aimed at extracting required fields in Vietnamese receipts captured by mobile devices :smile: We successfully built the flow and simply containerize as well as deploy the system on a webpage using streamlit. Everything is ready to use now!
The beblow image is the main pipeline of ours, which includes background subtraction(Maskrcnn), invoice alignment(craft+resnet34), text detection(craft), text recognition(vietocr) and key information extraction(graphsage). 

![image](https://user-images.githubusercontent.com/61444616/189092848-f3ea1b39-260c-479e-8b4e-61cbe55975b3.png)

About the dataset, we utilized [MC-OCR 2021](https://www.rivf2021-mc-ocr.vietnlp.com/). In general, the training set has 1155 images and the corresponding key fields, texts as the labels. Especially, this dataset is quite complex when having various backgrouds, as well as low quality images ... So EDA and proprcessing task are required to get good model performance!

More about Graphsage model, this is one of the popular graph-based model that can be used to handle node classification problem and in this case, node as the text box. In detail, Graphsage can be shortly described as ...
![image](https://user-images.githubusercontent.com/61444616/189104372-7f5c0ade-7f14-4532-813e-7d1a1ba4f9e1.png)


# Usage

You can easily run the project by running the below commands, but note that you already had docker in your computer. I 

```
git clone https://github.com/manhph2211/MC-OCR.git
docker build -t "app" .
docker run app
```

<!-- ![image](https://user-images.githubusercontent.com/61444616/189105320-9b78dff4-c1ed-467a-86c4-ea812496768b.png) -->

# Demo
![demo](https://user-images.githubusercontent.com/53470099/189409103-59ef12b7-ea3f-4170-b57b-5d24f59c24fb.gif)


# References

Thanks to the authors:

- [MaskRCNN](https://github.com/pytorch/vision)
- [DB](https://github.com/MhLiao/DB)
- [Vietocr](https://github.com/pbcquoc/vietocr)
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT)

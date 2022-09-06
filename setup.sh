echo "Setting up..."
pip install gdown
echo "Downloading checkpoints..."
gdown --fuzzy https://drive.google.com/file/d/1JWfHLMRwXVaZZCI0NpiL-VVooOYpv3TB/view?usp=sharing -O background_subtraction/maskrcnn/ckpts/model.pth
gdown --fuzzy https://drive.google.com/file/d/1pcwcYKeKl5l316m6xsoIgcNsw6mrGzre/view?usp=sharing -O text_detection/craft/ckpts/craft_mlt_25k.pth
gdown --fuzzy https://drive.google.com/file/d/1YoyKrJDEsGpMrCkgo9n44KfoD3VeRgJK/view?usp=sharing -O text_recognition/vietocr/ckpts/transformerocr.pth
# gdown --fuzzy https://drive.google.com/file/d/1JWfHLMRwXVaZZCI0NpiL-VVooOYpv3TB/view?usp=sharing -O background_subtraction/maskrcnn/ckpts/model.pth
echo "Loaded checkpoints"
echo "Running app..."
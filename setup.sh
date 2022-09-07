echo "Setting up..."
pip install gdown
pip install --upgrade streamlit
echo "Downloading checkpoints..."
gdown --fuzzy https://drive.google.com/file/d/1JWfHLMRwXVaZZCI0NpiL-VVooOYpv3TB/view?usp=sharing -O background_subtraction/maskrcnn/ckpts/model.pth
gdown --fuzzy https://drive.google.com/file/d/1pcwcYKeKl5l316m6xsoIgcNsw6mrGzre/view?usp=sharing -O text_detection/craft/ckpts/craft_mlt_25k.pth
gdown --fuzzy https://drive.google.com/file/d/1YoyKrJDEsGpMrCkgo9n44KfoD3VeRgJK/view?usp=sharing -O text_recognition/vietocr/ckpts/transformerocr.pth
gdown --fuzzy https://drive.google.com/file/d/1ZmoQY6-kURGWdF39ZnPwRzRJRqV-hcYL/view?usp=sharing -O key_info_extraction/ckpts/best_epoch2.pth
echo "Loaded checkpoints"
echo "Running app..."
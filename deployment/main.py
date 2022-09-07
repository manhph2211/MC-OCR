import sys
sys.path.append("/home/max/coding/MC-OCR")
from background_subtraction.maskrcnn.save_img import remove_bg 
from text_detection.craft.main import detect   
from text_recognition.main import recoginize
from key_info_extraction.tools.inference import get_key, visualize


def process_image(original_img_path="data/demo/original/mcocr_val_145114aszbc.jpg"):
    # original_img_path = "data/demo/original/mcocr_val_145114aszbc.jpg"
    # remove_bg(original_img_path)
    # detect(original_img_path.replace("original","bg_sub"))
    # remove_bg(original_img_path.replace("original","bg_sub"))
    # detect(original_img_path.replace("original","rotation"))
    # recoginize("data/demo/text_detection/data.json")
    # get_key("data/demo/text_recognition/data.json")
    # visualize("data/demo/kie/results.json")
    pass


if __name__ == "__main__":
    process_image()



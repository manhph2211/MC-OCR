from background_subtraction.maskrcnn.save_img import remove_background_one_image 
from text_detection.craft.main import detect   
from text_recognition.main import recoginize
from key_info_extraction.tools.inference import get_key


def main():
    original_img_path = "data/demo/original/mcocr_val_145114aszbc.jpg"
    remove_background_one_image(original_img_path)
    detect(original_img_path.replace("original","bg_sub"))
    remove_background_one_image(original_img_path.replace("original","bg_sub"))
    recoginize("data/demo/text_detection/data.json")
    get_key("data/demo/text_recognition/data.json")


if __name__ == "__main__":
    main()



import json
from vietocr.preprocessing import adaptive_threshold_gaussian, noise_removal
from vietocr.vietocr.tool.config import Cfg
from vietocr.vietocr.tool.predictor import Predictor


# Load Config for model
config = Cfg.load_config_from_name('vgg_transformer')
# Load weights. REMEMBER TO RESET WEIGHT PATH
config['weights'] = 'vietocr/ckpts/transformerocr.pth'
config['device'] = 'cpu'
detector = Predictor(config)


def infer_one_image(path_to_json_file = '../data/demo/text_detection/data.json'):
    # Define detector
    with open(path_to_json_file) as data_file:
        data = json.load(data_file)
        image_dir, file = list(data.items())[0]
        path_to_image = image_dir
        n_boxes = len(file)

        # Preprocess the image
        img = noise_removal(adaptive_threshold_gaussian(path_to_image[3:]))

        # Loop through the boxes:
        for box in range(n_boxes):
        # Get points of the cropped picture
            right = file[box]['crop'][0][0]
            bottom = file[box]['crop'][0][1]
            left = file[box]['crop'][1][0]
            top = file[box]['crop'][1][1]

            # Crop the considering image
            img_cropped = img.crop((left, top, right, bottom))

            # Infer
            prediction = detector.predict(img_cropped)
            
            # Match the result with the field in JSON file
            data[image_dir][box]['text'] = str(prediction)


    with open('C:\\Users\\manhph5\\Desktop\\RIVF2021-MC-OCR\\data\\demo\\text_recognition\\data.json', 'w') as fp:
        json.dump(data, fp, indent=2)

if __name__ == "__main__":
    infer_one_image()
annotation_path = 'data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/annotations/instances_default.json'
image_folder = 'data/mcocr2021_public_train_test_data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images'
mask_folder = 'data/mask'
export_data_path = 'data/data.json'
export_data_train_path = 'data/train.json'
export_data_val_path = 'data/val.json'
export_data_test_path = 'data/test.json'

#-------------------------------------------------

val_imgs = 'data/mcocr_public_train_test_shared_data/mcocr_val_data/val_images'
train_imgs = 'data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images'
save_train_img = 'data/train_images_after_semantic'
save_val_img = 'data/val_images_after_semantic'

#--------------------------------------------------

n_classes = 2
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'

model_save_path = 'background_subtraction/maskrcnn/ckpts/model.pth'
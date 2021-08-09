import config
from dataset import Receipt
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from local_utils import get_transform
import torch
from model import get_instance_segmentation_model


if __name__ == '__main__':
	# use our dataset and defined transformations
	dataset_train = Receipt(config.export_data_train_path, get_transform(train=True))
	dataset_val = Receipt(config.export_data_val_path, get_transform(train=False))
	dataset_test = Receipt(config.export_data_test_path, get_transform(train=False))

	# split the dataset in train and test set
	#torch.manual_seed(1)
	#indices = torch.randperm(len(dataset_train)).tolist()
	#dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
	#dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# define training and validation data loaders
	train_data_loader = torch.utils.data.DataLoader(
	    dataset_train, batch_size=2, shuffle=True, num_workers=2,
	    collate_fn=utils.collate_fn)

	val_data_loader = torch.utils.data.DataLoader(
	    dataset_val, batch_size=1, shuffle=True, num_workers=2,
	    collate_fn=utils.collate_fn)

	test_data_loader_test = torch.utils.data.DataLoader(
	    dataset_test, batch_size=1, shuffle=False, num_workers=2,
	    collate_fn=utils.collate_fn)

	device = config.device
	num_classes = config.n_classes
	# get the model using our helper function
	model = get_instance_segmentation_model(num_classes)
	# move model to the right device
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005,
	                            momentum=0.9, weight_decay=0.0005)

	# and a learning rate scheduler which decreases the learning rate by
	# 10x every 3 epochs
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
	                                               step_size=3,
	                                               gamma=0.1)

	# let's train it for 10 epochs
	num_epochs = 10

	for epoch in range(num_epochs):
	    # train for one epoch, printing every 10 iterations
	    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
	    # update the learning rate
	    lr_scheduler.step()
	    # evaluate on the test dataset
	    evaluate(model, val_data_loader, device=device)
	    torch.save(model.state_dict(), config.model_save_path)
import torch

from model.deeplabv3 import DeeplabV3
from dataset import ReceiptDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import os


def get_loader(image_size, batch_size, train_annotation='data/train.json', val_annotation='data/val.json'):
    train_dataset = ReceiptDataset(image_size, annotations=train_annotation)
    val_dataset = ReceiptDataset(image_size, annotations=val_annotation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader


def train_net(image_size=312, num_epochs=30, batch_size=16, save_checkpoint='./weights',
              load_checkpoint=None):
    train_loader, val_loader = get_loader(image_size=image_size, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    model = DeeplabV3(pretrained_backbone=True).to(device)
    if load_checkpoint is not None:
        model.load_state_dict(torch.load(load_checkpoint))
        print(f'Load checkpoint from: {load_checkpoint}')

    criterion = CrossEntropyLoss(size_average=True).to(device)
    # criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_losses = list()
    val_losses = list()
    best_val_loss = 999
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}: ')
        train_batch_losses = []
        val_batch_losses = []

        for image, mask in tqdm(train_loader):
            optimizer.zero_grad()
            with autocast():
                image = image.to(device)
                mask = mask.to(device)

                predict = model(image)
                loss = criterion(predict, mask)
            print(f'Batch loss: {loss.item()}')
            train_batch_losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_losses.append(sum(train_batch_losses) / len(train_batch_losses))
        print(f'=====================Train loss: {train_losses[-1]}===========================')

        for image, mask in tqdm(val_loader):
            image = image.to(device)
            mask = mask.to(device)

            predict = model(image)
            loss = criterion(predict, mask)
            val_batch_losses.append(loss.item())

        val_losses.append(sum(val_batch_losses) / len(val_batch_losses))
        print(f'=====================Validation loss: {val_losses[-1]}===========================')

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            if not os.path.isdir(save_checkpoint):
                os.makedirs(save_checkpoint)
            save_path = os.path.join(save_checkpoint, f'checkpoint{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    train_net()

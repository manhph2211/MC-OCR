import argparse
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from key_info_extraction.models.phobert_sage import SageNet
from key_info_extraction.datasets import Receipt
from key_info_extraction.loss import FocalLoss
from key_info_extraction.metrics import accuracy
from key_info_extraction.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PHOBERT-SAGE')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--val-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for validation (default: 4)')
    parser.add_argument('--train-json', type=str,
                        default="../../data/labeled_predict.json",
                        help='training folder path')
    parser.add_argument('--val-folder', type=str,
                        default="dataset/val_data/",
                        help='validation folder path')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='how many epoches to wait before saving model')
    parser.add_argument('--save-folder', type=str, default="logs/saved", metavar='N',
                        help='how many epoches to wait before saving model')
    parser.add_argument('--pretrain', type= str, default="logs/saved/model_best.pth",
                        help='Please enter pretrain')
    args = parser.parse_args()

    model = SageNet(in_channels=768)
    train_dataset = Receipt(json_file=args.train_json)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = None
    criterion = FocalLoss(gamma=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric = accuracy

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, metric, device='cuda')

    for epoch in tqdm(range(1, args.epochs + 1)):
        trainer.train(epoch)

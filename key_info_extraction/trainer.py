import torch


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, metric, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.device = device

    def train(self, epoch):
        self.model.train()

        losses = 0
        for step, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(data)
            loss = self.criterion(logits, data.y.reshape(-1))

            loss.backward()
            losses += loss.item()
            self.optimizer.step()

        print(f"Epoch {epoch}: Loss={losses}")

        return losses

    def val(self, epoch):
        self.model.val()

        accs = []
        losses = 0

        for step, data in enumerate(self.val_loader):
            data = data.to(self.device)

            logits = self.model(data)
            loss = self.criterion(logits, data.y.reshape(-1))
            losses += loss.item()
            preds = torch.argmax(logits, dim=1)

            accs.append(self.metric(preds, data.y))

        print(f"Epoch {epoch}: Loss={losses} - Accuracy:{sum(accs) / len(accs)}")

        return sum(accs) / len(accs)

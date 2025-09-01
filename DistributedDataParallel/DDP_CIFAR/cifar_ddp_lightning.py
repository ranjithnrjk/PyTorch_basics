import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from cifar_single_gpu import Simple_CNN
from torchmetrics.classification import MulticlassAccuracy

# -----------------
# Lightning Module
# -----------------

class CIFAR_10_classfier(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = Simple_CNN()
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.val_acc = MulticlassAccuracy(num_classes=10)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.train_acc(logits.softmax(dim=-1), y)
        self.log('train/acc', acc, on_step=False, on_epoch=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_acc(logits.softmax(dim=-1), y)
        self.log('val/acc', acc, on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class Cifar10DataModule(L.LightningDataModule):
    def __init__(self, data_dir:str='./data', batch_size:int=64, num_workers:int=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size=batch_size
        self.num_workers=num_workers

        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        if self.trainer.is_global_zero:
        # Called Only once
            datasets.CIFAR10(self.data_dir, train=True, download=True)
            datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self):
        # Called on every GPU
        full = datasets.CIFAR10(root='./data', train=True, transform=self.transform)
        self.train_data, self.val_data = random_split(full, [0.8, 0.2])
        self.test_data = datasets.CIFAR10(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
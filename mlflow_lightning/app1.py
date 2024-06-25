# Import Libraries
import torch
import os
import mlflow   
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from typing import Dict, Any

# Create a data module class
class MnistDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialization of Inherited LightningDataModule
        """
        super(MnistDataModule, self).__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.config = config

        # transforms for images
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        """
        Downloads the data, splits it into train, validation and test_sets
        :param stage: Stage - training or testing
        """
        self.df_train = datasets.MNIST('data', train=True, download=True, transform=self.transforms)

        self.df_train, self.df_val = random_split(self.df_train, [55000, 5000])

        self.df_test = datasets.MNIST('data', train=False, download=True, transform=self.transforms)


    def create_data_loader(self, df):
        """
        Generic dataloader function
        :param df: Input Tensor
        :return: returns the constructed dataloader
        """
        return DataLoader(
            df, batch_size=self.config['train']['batch_size'], 
            num_workers=self.config['train']['num_workers']
        )
    
    def train_dataloader(self):
        """
        : return : output - train dataloader for the given dataset
        """
        return self.create_data_loader(self.df_train)
    
    def val_dataloader(self):
        """
        :return: output - validation dataloader for the given dataset
        """
        return self.create_data_loader(self.df_val)

    def test_dataloader(self):
        """
        :return: output - test dataloader for the given dataset
        """
        return self.create_data_loader(self.df_test)

# Create a model class
class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        """
        :param config: Configuration dictionary containing:
            - model :
                - lr: float
                - batch_size: int
                - num_workers: int 
        """
        super(LightningMNISTClassifier, self).__init__()
        self.config = config

        # Mnist images are (1, 28, 28) (Channel, Height, Width)
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 10)

    
    
    def forward(self, x):
        """
        :param x: input features
        "return: output - mnist digit label for the input image
        """
        batch_size = x.size()[0]

        x = x.view(batch_size, -1) # (b, 1, 28, 28) -> (b, 1*28*28)

        # Layer 1 (b, 128) -> (b, 128)
        x = self.layer1(x)
        x = torch.relu(x)

        # Layer 2 (b, 128) -> (b, 256)
        x = self.layer2(x)
        x = torch.relu(x)

        # Layer 3 (b, 256) -> (b, 10)
        x = self.layer3(x)

        # Probability distribution over labels
        x = torch.log_softmax(x, dim=1)
        return x
    
    def cross_entropy_loss(self, logits, labels):
        """"
        Initializes the loss function
        :return: output - cross entropy loss
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        Trains the model on data in batches and returns the training loss on every batch

        :param train_batch: Input batch
        :param batch_idx: Batch index

        :return: output - training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'loss': loss}
    
    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches 

        :param val_batch: Input batch
        :param batch_idx: Batch index

        :return: output - validation loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        """
        Computes average validation loss over an epoch

        :param outputs: Output after every epoch end

        :return: output - average validation loss
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        Performs testing of data in batches

        :param test_batch: Input batch
        :param batch_idx: Batch index

        :return: output - test loss
        """
        x, y = test_batch
        outputs = self.forward(x)
        _, y_hat = torch.max(outputs, dim=1)
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        return {'test_acc': test_acc}
    
    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy over an epoch

        :param outputs: Output after every epoch end

        :return: output - average test accuracy
        """
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log('avg_test_acc', avg_test_acc, sync_dist=True)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return : output - optimizer and learning rate scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.config['lr'])

        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.config['scheduler_params']['mode'], 
                patience=self.config['scheduler_params']['patience'], 
                verbose=True, factor=self.config['scheduler_params']['factor'], 
                min_lr=1e-6
            ),
            'monitor': self.config['scheduler_params']['metric'],
        }
        return [self.optimizer], [self.scheduler]

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Lightning MNIST Example")

    parser.add_argument(
        "--es_monitor", type=str, default='val_loss', help='Early stopping monitor parameter'
    )

    parser.add_argument(
        "--es_mode", type=str, default='min', help='Early stopping mode parameter'
    )

    parser.add_argument(
        "--es_patience", type=int, default=2, help='Early stopping patience parameter'
    )

    parser.add_argument(
        "--es_verbose", type=bool, default=True, help='Early stopping verbose parameter'
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = LightningMNISTClassifier.add_model_specific_args(parent_parser=parser)

    mlflow.pytorch.autolog()

    args = parser.parse_args()
    dict_args = vars(args)

    if "accelerator" in dict_args:
        if dict_args["accelerator"] == "None":
            dict_args["accelerator"] = None

    model = LightningMNISTClassifier(**dict_args)

    dm = MnistDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage='fit')

    early_stopping = EarlyStopping(
        monitor=dict_args['es_monitor'],
        mode = dict_args['es_mode'],
        patience=dict_args['es_patience'],
        verbose=dict_args['es_verbose'],
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=""
    )

    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer.add_argparse_args(
        args, callbacks=[early_stopping, lr_logger], checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test()
# # Import Libraries
# import torch
# import yaml
# import mlflow
# from lightning import Trainer
# import lightning as L
# from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# from lightning.pytorch.loggers import MLFlowLogger
# import torchmetrics
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, datasets
# from typing import Dict, Any

# class MnistDataModule(L.LightningDataModule):
#     def __init__(self, config: Dict[str, Any]):
#         super(MnistDataModule, self).__init__()
#         self.config = config

#         # Transforms for images
#         self.transforms = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#         )

#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             mnist_full = datasets.MNIST('data', train=True, download=True, transform=self.transforms)
#             self.df_train, self.df_val = random_split(mnist_full, [55000, 5000])
        
#         if stage == 'test' or stage is None:
#             self.df_test = datasets.MNIST('data', train=False, download=True, transform=self.transforms)
    
#     def create_data_loader(self, dataset):
#         return DataLoader(
#             dataset, batch_size=self.config['train']['batch_size'], 
#             num_workers=self.config['train']['num_workers']
#         )
    
#     def train_dataloader(self):
#         return self.create_data_loader(self.df_train)
    
#     def val_dataloader(self):
#         return self.create_data_loader(self.df_val)

#     def test_dataloader(self):
#         return self.create_data_loader(self.df_test)


# # Create a model class
# class LightningMNISTClassifier(L.LightningModule):
#     def __init__(self, config: Dict[str, Any]):
#         """
#         :param config: Configuration dictionary containing:
#             - model :
#                 - lr: float
#                 - batch_size: int
#                 - num_workers: int 
#         """
#         super(LightningMNISTClassifier, self).__init__()
#         self.config = config

#         # Mnist images are (1, 28, 28) (Channel, Height, Width)
#         self.layer1 = torch.nn.Linear(28 * 28, 128)
#         self.layer2 = torch.nn.Linear(128, 256)
#         self.layer3 = torch.nn.Linear(256, 10)

#     def forward(self, x):
#         """
#         :param x: input features
#         :return: output - mnist digit label for the input image
#         """
#         batch_size = x.size()[0]

#         x = x.view(batch_size, -1)  # (b, 1, 28, 28) -> (b, 1*28*28)

#         # Layer 1 (b, 128) -> (b, 128)
#         x = self.layer1(x)
#         x = torch.relu(x)

#         # Layer 2 (b, 128) -> (b, 256)
#         x = self.layer2(x)
#         x = torch.relu(x)

#         # Layer 3 (b, 256) -> (b, 10)
#         x = self.layer3(x)

#         # Probability distribution over labels
#         x = torch.log_softmax(x, dim=1)
#         return x

#     def cross_entropy_loss(self, logits, labels):
#         """"
#         Initializes the loss function
#         :return: output - cross entropy loss
#         """
#         return F.nll_loss(logits, labels)

#     def training_step(self, train_batch, batch_idx):
#         """
#         Trains the model on data in batches and returns the training loss on every batch

#         :param train_batch: Input batch
#         :param batch_idx: Batch index

#         :return: output - training loss
#         """
#         x, y = train_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)

#         # Calculate accuracy
#         _, predicted = torch.max(logits, 1)
#         acc = (predicted == y).float().mean()
        
#         # Log metrics
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
#         return loss

#     def on_train_epoch_end(self):
#         train_loss = self.trainer.callback_metrics['train_loss_epoch']
#         train_acc = self.trainer.callback_metrics['train_acc_epoch']
#         val_loss = self.trainer.callback_metrics['val_loss_epoch']
#         val_acc = self.trainer.callback_metrics['val_acc_epoch']

#         epoch = self.current_epoch

#         log_text = f"""Epoch : {epoch}
#         Training Loss : {train_loss:.4f} | Training Accuracy : {train_acc:.4f}
#         Validation Loss : {val_loss:.4f} |Validation Accuracy : {val_acc:.4f}
#         """
#         # Log the text
#         mlflow.log_text(log_text, f"epoch_{epoch}_log.txt")

#         print(log_text)

#     def validation_step(self, val_batch, batch_idx):
#         """
#         Performs validation of data in batches 

#         :param val_batch: Input batch
#         :param batch_idx: Batch index

#         :return: output - validation loss
#         """
#         x, y = val_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)

#         # Calculate accuracy
#         _, predicted = torch.max(logits, 1)
#         acc = (predicted == y).float().mean()
        
#         # Log metrics
#         self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
 
#         return {'val_loss': loss, 'val_acc': acc}

#     def on_validation_epoch_end(self):
#         """
#         Computes average validation loss over an epoch
#         """
#         if 'val_loss' in self.trainer.callback_metrics:
#             avg_loss = self.trainer.callback_metrics['val_loss']
#             # Logging the average validation loss
#             self.log('avg_val_loss', avg_loss, sync_dist=True) 
#         else:
#             print("val_loss not found in callback_metrics")

#     def test_step(self, test_batch, batch_idx):
#         """
#         Performs testing of data in batches

#         :param test_batch: Input batch
#         :param batch_idx: Batch index

#         :return: output - test loss
#         """
#         x, y = test_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         _, y_hat = torch.max(logits, dim=1)
#         accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
#         test_acc = accuracy(y_hat.cpu(), y.cpu())
        
#         # Logging the test loss and test accuracy
#         self.log('test_loss', loss, prog_bar=True)
#         self.log('test_acc', test_acc, prog_bar=True)
#         return {'test_loss': loss, 'test_acc': test_acc}

#     def on_test_epoch_end(self):
#         """
#         Computes average test loss over an epoch
#         """
#         if 'test_loss' in self.trainer.callback_metrics:
#             avg_test_loss = self.trainer.callback_metrics['test_loss']
#             # Logging the average test loss
#             self.log('avg_test_loss', avg_test_loss, sync_dist=True) 
#         else:
#             print("test_loss not found in callback_metrics")

#     def configure_optimizers(self):
#         """
#         Initializes the optimizer and learning rate scheduler

#         :return: output - optimizer and learning rate scheduler
#         """
#         optimizer = torch.optim.Adam(self.parameters(), 
#                                      lr=self.config['optimizer_params']['learning_rate'])

#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
#         factor=0.1, patience=2, verbose=True)
        
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'monitor': self.config['scheduler_params']['metric'],
#                 'interval': 'epoch',
#                 'frequency': 1,
#             }
#         }
    
#     def on_fit_start(self):
#         mlflow.pytorch.autolog()
    


# if __name__ == "__main__":

#     # Load configuration from YAML file
#     with open('config.yaml', 'r') as file:
#         config = yaml.safe_load(file)

#     dm = MnistDataModule(config)
#     model = LightningMNISTClassifier(config)

#     early_stopping = EarlyStopping(
#         monitor=config['train']['es_monitor'],
#         mode=config['train']['es_mode'],
#         patience=config['train']['es_patience'],
#         verbose=config['train']['es_verbose'],
#     )

#     checkpoint_callback = ModelCheckpoint(
#         monitor='val_loss',
#         dirpath='models',
#         filename='mnist-{epoch:02d}-{val_loss:.2f}',
#         save_top_k=3,
#         mode='min',
#     )
    
#     mlf_logger = MLFlowLogger(experiment_name="MNIST_Experiment")

#     lr_monitor = LearningRateMonitor(logging_interval='epoch')

#     trainer = Trainer(
#         max_epochs=3,
#         callbacks=[checkpoint_callback, lr_monitor, early_stopping],
#         logger=mlf_logger
#     )

#     trainer.fit(model, datamodule=dm)
#     trainer.test(model, datamodule=dm)






# Import Libraries
import torch
import yaml
import mlflow
from lightning import Trainer
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from typing import Dict, Any

class MnistDataModule(L.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super(MnistDataModule, self).__init__()
        self.config = config

        # Transforms for images
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST('data', train=True, download=True, transform=self.transforms)
            self.df_train, self.df_val = random_split(mnist_full, [55000, 5000])
        
        if stage == 'test' or stage is None:
            self.df_test = datasets.MNIST('data', train=False, download=True, transform=self.transforms)
    
    def create_data_loader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.config['train']['batch_size'], 
            num_workers=self.config['train']['num_workers']
        )
    
    def train_dataloader(self):
        return self.create_data_loader(self.df_train)
    
    def val_dataloader(self):
        return self.create_data_loader(self.df_val)

    def test_dataloader(self):
        return self.create_data_loader(self.df_test)


# Create a model class
class LightningMNISTClassifier(L.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super(LightningMNISTClassifier, self).__init__()
        self.config = config

        # Mnist images are (1, 28, 28) (Channel, Height, Width)
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)  # (b, 1, 28, 28) -> (b, 1*28*28)

        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        _, predicted = torch.max(logits, 1)
        acc = (predicted == y).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        train_acc = self.trainer.callback_metrics.get('train_acc_epoch', 0.0)
        val_loss = self.trainer.callback_metrics.get('val_loss_epoch', 0.0)
        val_acc = self.trainer.callback_metrics.get('val_acc_epoch', 0.0)
        epoch = self.current_epoch

        log_text = f"""Epoch : {epoch}\n
        Training Loss : {train_loss:.4f} | Training Accuracy : {train_acc:.4f}\n
        Validation Loss : {val_loss:.4f} | Validation Accuracy : {val_acc:.4f}\n
        """

        # Write logs to a file
        log_file_path = f"epoch_{epoch}_log.txt"
        with open(log_file_path, "w") as log_file:
            log_file.write(log_text)

        # Log the file to MLflow
        mlflow.log_artifact(log_file_path)

        print(log_text)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        _, predicted = torch.max(logits, 1)
        acc = (predicted == y).float().mean()
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
 
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        self.log('avg_val_loss', avg_loss, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        _, y_hat = torch.max(logits, dim=1)
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', test_acc, prog_bar=True)
        return {'test_loss': loss, 'test_acc': test_acc}

    def on_test_epoch_end(self):
        avg_test_loss = self.trainer.callback_metrics.get('test_loss', 0.0)
        self.log('avg_test_loss', avg_test_loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['optimizer_params']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
        factor=0.1, patience=2, verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.config['scheduler_params']['metric'],
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def on_fit_start(self):
        mlflow.pytorch.autolog()

if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    dm = MnistDataModule(config)
    model = LightningMNISTClassifier(config)

    early_stopping = EarlyStopping(
        monitor=config['train']['es_monitor'],
        mode=config['train']['es_mode'],
        patience=config['train']['es_patience'],
        verbose=config['train']['es_verbose'],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='models',
        filename='mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    mlf_logger = MLFlowLogger(experiment_name="MNIST_Experiment")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=3,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=mlf_logger
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

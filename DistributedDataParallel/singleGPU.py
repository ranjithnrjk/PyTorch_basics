import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataUtils import MyTrainDataset

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int,
                 save_every: int) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()

    def _run_epochs(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"GPU {self.gpu_id} Epoch {epoch} | Batchsize {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = 'checkpoint.pt'
        torch.save(ckp, PATH)
        print(f"Epoch: {epoch} | Trainin checkpoint saved at epoch {epoch}")

    def train(self, max_epochs:int):
        for epoch in range(max_epochs):
            self._run_epochs(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2000) # Load training data
    model = torch.nn.Linear(20, 1) # Load a single linear layer for simplicity
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True)


def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    dataloader = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, dataloader, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Simple distributed training job.')
    parser.add_argument('total_epochs', type=int, help='Number of epoch to train the model.')
    parser.add_argument('save_every', type=int, help='How often the model needs to be saved.')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size on each device (default:32)')
    args = parser.parse_args()

    device = 0 # Shorthand for cuda
    main(device, args.total_epochs, args.save_every, args.batch_size)
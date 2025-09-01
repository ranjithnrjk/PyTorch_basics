import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataUtils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_process_group(backend='nccl')    

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 save_every: int,
                 snapshot_path: str) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists('snapshot.pt'):
            print('Loading Snapshot...')
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f'Resuming training from snapshot at epoch: {self.epochs_run}')

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
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        PATH = 'snapshot.pt'
        torch.save(snapshot, PATH)
        print(f"Epoch: {epoch} | Training snapshot saved at snapshot.pt")

    def train(self, max_epochs:int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epochs(epoch)
            if self.gpu_id ==0 and epoch % self.save_every == 0:
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
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset))


def main(total_epochs: int, save_every: int, batch_size: int, snapshot_path: str='snapshot.pt'):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Simple distributed training job.')
    parser.add_argument('total_epochs', type=int, help='Number of epoch to train the model.')
    parser.add_argument('save_every', type=int, help='How often the model needs to be saved.')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size on each device (default:32)')
    args = parser.parse_args()
    main(args.total_epochs, args.save_every, args.batch_size)
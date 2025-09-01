import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# The model is defined in the single gpu training code and importing it for simplicity
from cifar_single_gpu import Simple_CNN

#-------------
# 🔌 DDP Setup
#-------------

def setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend='nccl')

def clean_up():
    dist.destroy_process_group()

# -------------------------------
# Data loaders
# -------------------------------
def get_dataloaders(rank, world_size, batch_size=64):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_sampler = DistributedSampler(dataset=trainset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(dataset=valset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader, train_sampler


# -------------------
# Validation Function
# -------------------
def validation(model, val_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# -----------------------
# Train one epoch
# -----------------------
def train_one_epoch(model, device, optimizer, criterion, trainloader, epoch, rank):

    epoch_loss = 0.0
    model.train()

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 100 == 0 and rank == 0:
            print(f"Epoch: {epoch+1}: Loss at Batch: {i} -> {loss.item():.4f}")

    return epoch_loss / len(trainloader)


# -------------------------------
# 🚂 Main function
# -------------------------------
def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    setup()
    device = torch.device(f"cuda:{rank}")

    model = Simple_CNN().to(device)
    model = DDP(module=model, device_ids=[rank])

    train_loader, val_loader, train_sampler = get_dataloaders(rank, world_size=world_size, batch_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_accuracy = 0
    for epoch in range(10):
        train_sampler.set_epoch(epoch
                                )
        loss = train_one_epoch(model, device, optimizer, criterion, train_loader, epoch, rank)
        
        if rank == 0:
            val_acc = validation(model, val_loader, device)

            print(f"Loss at the end of {epoch+1} epoch -> {loss:.4f}: Val Accuracy: {val_acc:.2f} %")

            os.makedirs('./checkpoints', exist_ok=True)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'val_acc': val_acc
                }, f"checkpoints/checkpoint_epoch_{epoch+1}_acc_{best_val_accuracy}.pt")
    
    if rank == 0:
        print(f'Training complete!\nTotal Loss:{loss:.4f}')

    clean_up()

        

# Entry point
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")
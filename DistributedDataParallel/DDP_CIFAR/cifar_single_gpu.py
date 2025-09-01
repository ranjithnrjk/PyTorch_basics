import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# -----------------------
# 🧠 Simple CNN
# -----------------------
class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x))) # [B, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------
# 🚀 Training loop
# -----------------------
def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    model = Simple_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        running_loss = 0.0
        for i, (inputs,labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print(f"[Epoch {epoch+1}] Batch {i}, Loss: {loss.item():.4f}")
    print("Training complete!")
    print(f"Loss: {running_loss}")

if __name__ == "__main__":
    import time
    start = time.time()
    train()
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")
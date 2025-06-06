import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch)
            predicted_labels = (y_pred > 0.5).float()
            predicted_labels = torch.argmax(predicted_labels, dim=1)
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

def plot_predictions(model, dataloader):
    model.eval()
    X_batch, y_batch = next(iter(dataloader))
    with torch.no_grad():
        y_pred = model(X_batch)
        predicted_labels = (y_pred > 0.5).float()
        predicted_labels = torch.argmax(predicted_labels, dim=1)

    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(X_batch[i][0], cmap="gray")
        plt.title(f"True: {y_batch[i]}\nPred: {predicted_labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


full_train_dataset = datasets.FashionMNIST(
    root="Data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

trainDataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valDataLoader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = datasets.FashionMNIST(
    root="Data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
testDataLoader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = ImageClassifier()
lossFunction = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 20
for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0
    for X_batch, y_batch in trainDataLoader:
        y_pred = model(X_batch)
        y_batch_onehot = F.one_hot(y_batch, num_classes=10).float()

        loss = lossFunction(y_pred, y_batch_onehot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in valDataLoader:
            y_pred = model(X_batch)
            y_batch_onehot = F.one_hot(y_batch, num_classes=10).float()
            loss = lossFunction(y_pred, y_batch_onehot)
            val_loss += loss.item()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss/len(trainDataLoader):.4f}, Val Loss: {val_loss/len(valDataLoader):.4f}")


test_acc = accuracy(model, testDataLoader)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


plot_predictions(model, testDataLoader)

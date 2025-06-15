import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import numpy

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.blockOne = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, 
                out_channels = 16, 
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2
            )
        )
        self.blockTwo = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 16 * 7 * 7, 
                      out_features = 10)
        )

    def forward(self, tnsr : torch.Tensor) -> torch.Tensor:
        tnsr = self.blockOne(tnsr)
        tnsr = self.blockTwo(tnsr)
        tnsr = self.classifier(tnsr)
        return tnsr
def plot_predictions(model, dataset):
    model.eval()
    rand_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    X_batch, y_batch = next(iter(rand_loader))
    with torch.no_grad():
        y_pred = model(X_batch)
        predicted_labels = torch.argmax(y_pred, dim=1)
    classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"]
    plt.figure(figsize=(10, 10))
    for i in range(32):
        plt.subplot(8, 4, i+1)
        plt.imshow(X_batch[i][0], cmap="gray")
        true_label = classes[y_batch[i].item()]
        pred_label = classes[predicted_labels[i].item()]
        color = 'green' if y_batch[i] == predicted_labels[i] else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=7, color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
trainData = datasets.FashionMNIST(
    root = "Data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = None
)
testData = datasets.FashionMNIST(
    root = "Data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = None
)
trainDataLoader = DataLoader(
    dataset = trainData,
    batch_size = 32,
    shuffle = True
)
testDataLoader = DataLoader(
    dataset = testData,
    batch_size = 32,
    shuffle = False
)
classes = testData.classes
model = ImageClassifier()
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)
epochs = 5
totalSteps = epochs * len(trainDataLoader)
losses = []
with tqdm(total = totalSteps, desc = "Training Progress") as pbar:
    for epoch in range(epochs):
        model.train()
        trainLossBatch = 0
        for X, y in trainDataLoader:
            yTrainPred = model(X)
            trainLoss = lossFunction(yTrainPred, y)
            trainLossBatch += trainLoss.item()
            optimizer.zero_grad()
            trainLoss.backward()
            optimizer.step()
            pbar.update(1)
        averageLoss = trainLossBatch / len(trainDataLoader)
        losses.append(f"Epoch {epoch + 1}/{epochs} - Loss: {averageLoss:.4f}")
for _ in losses:
    print(_)
yPredList = []
model.eval()
with torch.inference_mode():
    for X, y in tqdm(testDataLoader, desc="Making Predictions"):
        yTestLogitPred = model(X)
        yPred = torch.argmax(yTestLogitPred, dim = 1)
        yPredList.append(yPred)
yPredTensor = torch.cat(yPredList)
plot_predictions(model = model, dataset = trainData)
confusionMatrix = ConfusionMatrix(task="multiclass", num_classes=len(classes))
confusionMatrixTensor = confusionMatrix(
    preds = yPredTensor,
    target = testData.targets
)
figure, axis = plot_confusion_matrix(
    conf_mat = confusionMatrixTensor.numpy(),
    class_names = classes,
    figsize = (10, 10)
)
plt.show()

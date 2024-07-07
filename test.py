import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             classification_report)
import numpy as np
from model_cifar10 import CNN

# Define classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Define data transform for the test set
transform_test = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image
])

# Load test dataset
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                         transform=transform_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define model structure
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # First convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Second convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # First max pooling layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Third convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Fourth convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second max pooling layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Fifth convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Sixth convolutional layer
            nn.BatchNorm2d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Third max pooling layer
            nn.Dropout(0.5),  # Dropout layer with probability 0.5
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),  # Fully connected layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Linear(512, num_classes),  # Fully connected layer
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through the feature extractor
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.classifier(x)  # Pass flattened output through the classifier
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=len(classes)).to(device)
model = torch.load("CIFAR_model.pth", map_location=device)  # Load the saved model
model.eval()  # Set the model to evaluation mode

# Evaluate model
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():  # Disable gradient calculation
    for data in test_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)  # Forward pass
        _, preds = torch.max(outputs, 1)  # Get the predicted class
        all_labels.extend(targets.cpu().numpy())  # Store the true labels
        all_preds.extend(preds.cpu().numpy())  # Store the predicted labels
        all_probs.extend(outputs.cpu().numpy())  # Store the output probabilities

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

# Calculate ROC AUC and PRC AUC for each class
roc_auc = {}
prc_auc = {}
for i in range(10):
    y_true = np.array(all_labels) == i  # True labels for class i
    y_score = np.array(all_probs)[:, i]  # Predicted scores for class i
    roc_auc[classes[i]] = roc_auc_score(y_true, y_score)  # Calculate ROC AUC
    precision_, recall_, _ = precision_recall_curve(y_true, y_score)  # Calculate precision-recall curve
    prc_auc[classes[i]] = np.trapz(recall_, precision_)  # Calculate PRC AUC

# Print metrics
print('Accuracy: {:.4f}'.format(accuracy))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1))
print('Confusion Matrix:')
print(conf_matrix)

# Print detailed classification report
print('Classification Report:')
print(classification_report(all_labels, all_preds, target_names=classes))

# Print ROC and PRC AUC for each cla+ss
print('ROC AUC and PRC AUC for each class:')
for i in range(10):
    print('{} - ROC AUC: {:.4f}, PRC AUC: {:.4f}'.format(classes[i], roc_auc[classes[i]], prc_auc[classes[i]]))

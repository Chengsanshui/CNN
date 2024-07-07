import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # First convolutional layer (input channels: 3, output channels: 128)
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Second convolutional layer (input channels: 128, output channels: 128)
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # First max pooling layer (kernel size: 2, stride: 2)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Third convolutional layer (input channels: 128, output channels: 256)
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Fourth convolutional layer (input channels: 256, output channels: 256)
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second max pooling layer (kernel size: 2, stride: 2)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Fifth convolutional layer (input channels: 256, output channels: 512)
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Sixth convolutional layer (input channels: 512, output channels: 512)
            nn.BatchNorm2d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Third max pooling layer (kernel size: 2, stride: 2)
            nn.Dropout(0.5),  # Dropout layer with a dropout probability of 0.5
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),  # Fully connected layer (input features: 512*4*4, output features: 512)
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Linear(512, num_classes),  # Fully connected layer (input features: 512, output features: num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through the feature extraction layers
        x = x.view(x.size(0), -1)  # Flatten the output from the feature extraction layers
        x = self.classifier(x)  # Pass the flattened output through the classifier
        return x






if __name__ == '__main__':
    model = CNN()  # Instantiate the CNN model
    input_tensor = torch.ones((64, 3, 32, 32))  # Create a dummy input tensor (batch size: 64, channels: 3, height: 32, width: 32)
    output_tensor = model(input_tensor)  # Pass the input tensor through the model
    print(output_tensor.shape)  # Print the shape of the output tensor



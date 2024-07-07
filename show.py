import torch
import torchvision.transforms
from torch import nn
from PIL import Image

# Path to the test image
image_path = "./test_imgs./1.png"
# Open the image using PIL
image = Image.open(image_path)
print(image)

# Define the classes in the dataset
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# Define the transformation: resize the image to 32x32 and convert it to a tensor
tranform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

# Apply the transformation to the image
image = tranform(image)
print(image.shape)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # Convolutional layer with 32 filters, kernel size 5x5
            nn.MaxPool2d(2),            # Max pooling with kernel size 2x2
            nn.ReLU(inplace=True),      # ReLU activation
            nn.Conv2d(32, 32, 5, 1, 2), # Convolutional layer with 32 filters, kernel size 5x5
            nn.MaxPool2d(2),            # Max pooling with kernel size 2x2
            nn.ReLU(inplace=True),      # ReLU activation
            nn.Conv2d(32, 64, 5, 1, 2), # Convolutional layer with 64 filters, kernel size 5x5
            nn.MaxPool2d(2),            # Max pooling with kernel size 2x2
            nn.ReLU(inplace=True),      # ReLU activation
            nn.Conv2d(64, 128, 5, 1, 2),# Convolutional layer with 128 filters, kernel size 5x5
            nn.MaxPool2d(2),            # Max pooling with kernel size 2x2
            nn.ReLU(inplace=True),      # ReLU activation
            nn.Conv2d(128, 128, 5, 1, 2),# Convolutional layer with 128 filters, kernel size 5x5
            nn.MaxPool2d(2),            # Max pooling with kernel size 2x2
            nn.ReLU(inplace=True),      # ReLU activation
            nn.Flatten(),               # Flatten the tensor
            nn.Linear(128 * 4 * 4, 128),# Fully connected layer with 128 units
            nn.Linear(128, 10)          # Output layer with 10 units (one for each class)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Select the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = torch.load("CIFAR_model.pth", map_location=device)
print(model)

# Set the model to evaluation mode
model.eval()

# Move the image tensor to the selected device
image = image.to(device)

# Reshape the image tensor to fit the model input (batch_size, channels, height, width)
image = torch.reshape(image, (1, 3, 32, 32))

# Disable gradient calculation for inference
with torch.no_grad():
    # Get the model output
    output = model(image)
    print(output)

# Get the predicted class index
print(output.argmax(1))

# Get the class label from the predicted index
predicted = torch.argmax(output, 1)
print("predicted: ", classes[predicted.item()])

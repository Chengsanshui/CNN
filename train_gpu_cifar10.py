import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_cifar10 import *  # Make sure to replace this with the actual path to your model definition
import torch
import torchvision
from torchvision import transforms

# Define the device to train the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Prepare the dataset with transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image
])

# Load the CIFAR-10 dataset
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=transform_train,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transform_test,
                                         download=True)

# Print the length of the train and test datasets
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Length of train dataset: {}".format(train_data_size))
print("Length of test dataset: {}".format(test_data_size))

# Use DataLoader to load the dataset
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Create a network model
# The model is imported from python file called model_cifar10
cnn = CNN().to(device)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# Define the optimizer
learning_rate = 1e-3
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

# Set some parameters for the training process
total_train_step = 0  # Record the number of training steps
total_test_step = 0  # Record the number of testing steps
epoch = 101  # Number of training epochs

# Add TensorBoard to log the training process
writer = SummaryWriter("cifar10/logs1")
start_time = time.time()

# Start the training process
for i in range(epoch):
    print("------The {} round is starting------".format(i + 1))
    cnn.train()  # Set the model to training mode
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = cnn(imgs)  # Forward pass
        loss = loss_fn(outputs, targets)  # Compute the loss
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        total_train_step += 1

        if total_train_step % 100 == 0:
            end_time = time.time()
            print("Time: {}s".format(end_time - start_time))
            print("Number of training steps: {}, Loss: {}".format(total_train_step,
                                                                  loss.item()))
            print("-----------------------------------")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # Start the testing process
    cnn.eval()  # Set the model to evaluation mode
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # Disable gradient calculation
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = cnn(imgs)  # Forward pass
            loss = loss_fn(outputs, targets)  # Compute the loss
            total_test_loss += loss.item()  # Accumulate the test loss
            accuracy = (outputs.argmax(1) == targets).sum()  # Compute the accuracy
            total_accuracy += accuracy.item()

    print("------------------------------------------------------------------")
    print("Total loss of test dataset: {}".format(total_test_loss))
    print("Total accuracy of test dataset: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # Save the model every 10 epochs
    if i % 10 == 0:
        torch.save(cnn, "CIFAR10_{}.pth".format(i))
        print("Model has been saved")
writer.close()

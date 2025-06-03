import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
 
# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])
 
# Download the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
 
# Create data loaders to load the data in batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
         
        # Define the layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Conv layer: 1 input channel (grayscale), 32 output channels
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(32 * 6 * 6, 10)  # Fully connected layer
 
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply conv, ReLU activation, and pooling
        x = x.view(-1, 32 * 6 * 6)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)  # Apply the fully connected layer
        return x
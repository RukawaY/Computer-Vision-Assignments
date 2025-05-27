import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2) # Input: 1x28x28, Output: 6x28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # Output: 6x14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0) # Output: 16x10x10
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # Output: 16x5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
    

def visualize_images(trainloader, testloader):
    # randomly select 16 images and show as a 4x4 grid
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for row in range(4):
            for col in range(4):
                axs[row, col].imshow(images[row * 4 + col].numpy().squeeze(), cmap='gray')
                axs[row, col].axis('off') # Hide axis
        plt.savefig("train_images.png")
        break

    for i, data in enumerate(testloader, 0):
        images, labels = data
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for row in range(4):
            for col in range(4):
                axs[row, col].imshow(images[row * 4 + col].numpy().squeeze(), cmap='gray')
                axs[row, col].axis('off') # Hide axis
        plt.savefig("test_images.png")
        break

# Training loop
def train(epochs):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            if i % 200 == 199: # Print every 200 mini-batches
                logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f} accuracy: {100 * correct / total:.3f}')
                running_loss = 0.0
        logger.info(f'Epoch {epoch + 1} completed. Total loss: {running_loss:.3f} Total accuracy: {100 * correct / total:.3f}')
    logger.info('Finished Training')

# Evaluation function
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Evaluating"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logger.info(f'Accuracy of the network on the 10000 test images: {accuracy:.3f} %')
    return accuracy

# show 16 images and their predictions
def evaluate_visualize():
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            selected_images = images[0:16].to("cpu")
            selected_predictions = predicted[0:16].to("cpu")
            fig, axs = plt.subplots(4, 4, figsize=(10, 10))
            for row in range(4):
                for col in range(4):
                    axs[row, col].imshow(selected_images[row * 4 + col].numpy().squeeze(), cmap='gray')
                    axs[row, col].set_title(f'Predicted: {selected_predictions[row * 4 + col]}')
                    axs[row, col].axis('off') # Hide axis
            plt.savefig("predictions.png")
            break


if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and preprocess MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)

    # Visualize training and test images
    visualize_images(trainloader, testloader)

    # Initialize model, loss function, and optimizer
    model = LeNet5()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10 # Define the number of epochs for training
    train(num_epochs)
    evaluate()
    evaluate_visualize()

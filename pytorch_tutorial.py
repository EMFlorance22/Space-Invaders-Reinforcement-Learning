import torch
import torchvision # We get the data from torchvision and transform it into a tensor
import torchvision.transforms as transforms

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # We transform whatever image data we want into a normalized Tensor

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform) # Import the CIFAR10 data and transform it
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2) # Load the training data using torch's DataLoader
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform) # Import the testing data and transform it
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # We have a tuple of the image classes the Classifier will use

import torch.nn as nn # This allows us to create a neural network in pytorch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # Function that allows us to propagate forward in the NN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# We use a Classification Cross-Entropy Loss Function and SGD with momentum
import torch.optim as optim

criterion = nn.CrossEntropyLoss() # Define the loss function for the neural network
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # The optimizer used to train the network

for epoch in range(2):  # loop over the dataset multiple times and train the model

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data # Inputs are the images/tensor object, labels are the classes

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # Propagate the input tensors/images through a CNN
        loss = criterion(outputs, labels) # Compute the loss of the outputs and class labels
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Testing the Trained Neural Network

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# Function to show an image in the CIFAR10 Dataset
def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy() # Turn the image into a numpy array?
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # Use the matplotlib method imshow to show this image
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

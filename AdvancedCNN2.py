import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class Net(nn.Module):
    # Use this function to define your network
    # Creates the network
    def __init__(self):
        super().__init__()
        # Inits the model layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Defines forward apth
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_classifier():
    #Define Transfrom
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #Define batch size
    batch_size = 4
    #Define dataset and dataloader
    trainset = torchvision.datasets.CIFAR10(root='./cifar10/train', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10/test', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    #Define classes
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Creates Network 
    net = Net()


    # Defines loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(2):  # loop over the dataset for 2 iteration
        #pass
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

           

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

    # Saves the model weights after training
    PATH = './cifar10/cifar_net_2epoch.pth'
    torch.save(net.state_dict(), PATH)
    return net

def evalNetwork():
    # Initialized the network and load from the saved weights
    PATH = './cifar10/cifar_net_2epoch.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    # Loads dataset
    batch_size= 4
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./cifar10/test', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # Evaluates samples
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def weight_visualization_ab(q1=True):
    # Display the weights of the convolution kernels from Question #5 of the previous HW 4. 
    # Ignore bias. Please include the results in your report.pdf, 
    # the question will be graded manually. 
    # Feel free use other code structure
    # Please include the plots to pdf report
    if q1:
        net = Net()
        state_dict = torch.load('./cifar10/cifar_net_2epoch.pth')
        net.load_state_dict(state_dict)
        weights = net.state_dict()['conv1.weight'].numpy()

        # Normalize weights
        min_val = np.min(weights)
        max_val = np.max(weights)
        weights = (weights - min_val) / (max_val - min_val)

        # Display the weights as colored image patches
        fig, axs = plt.subplots(1, weights.shape[0], figsize=(20, 2))
        for i, ax in enumerate(axs):
            ax.imshow(np.transpose(weights[i], (1, 2, 0)))
            axs[i].set_title(f'Color channel {i+1}')
            ax.axis('off')
        plt.show()

        # Convert the weights to grayscale and display them as grayscale image patches
        color_channels = ['Red', 'Green', 'Blue']
        fig, axs = plt.subplots(3, weights.shape[0], figsize=(20, 2 * 3))
        for i in range(3):  # 3 color channels
            for j in range(weights.shape[0]):
                axs[i, j].imshow(weights[j, i], cmap='gray')
                axs[i, j].set_title(f'{color_channels[i]} channel {j+1}')
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.show()

    else:
        net = Net()
        state_dict = torch.load('./cifar10/cifar_net_2epoch.pth')
        net.load_state_dict(state_dict)

        second_layer_weights = list(net.children())[2].weight.data.numpy()
        print(second_layer_weights.shape)

        # Normalize weights
        min_val = np.min(second_layer_weights)
        max_val = np.max(second_layer_weights)
        second_layer_weights = (second_layer_weights - min_val) / (max_val - min_val)

        #Channels
        incoming_channels = second_layer_weights.shape[1]
        outgoing_channels = second_layer_weights.shape[0]

        fig, axs = plt.subplots(outgoing_channels,incoming_channels,  figsize=(20, 2 * 3))
        for i in range(outgoing_channels):  # 16 outgoing channels
            for j in range(incoming_channels): # 6 incoming channels
                axs[i, j].imshow(second_layer_weights[i, j], cmap='gray')
                axs[i, j].set_title(f'Out: {i+1} In: {j+1}',fontsize=8)
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.show()

def hypaparameter_sweep():

    #Define Transfrom
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #Define batch size
    batch_size = 4
    #Define dataset and dataloader
    trainset = torchvision.datasets.CIFAR10(root='./cifar10/train', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10/test', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    #Define classes
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Creates Network 

    learning_rates = [0.01, 0.001, 0.0001]
    losses = {lr: [] for lr in learning_rates}
    train_errors = {lr: [] for lr in learning_rates}
    test_errors = {lr: [] for lr in learning_rates}

    for lr in learning_rates:
        net = Net()

        # Defines loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        for epoch in range(2):  # loop over the dataset for 2 iteration
            #pass
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

            

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                # append the current loss to the training losses list

                running_loss += loss.item()

                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'LR={lr} [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    losses[lr].append(running_loss/ 2000)
                    running_loss = 0.0

                    # compute accuracy on a random sample of 1000 training images
                    train_error = round((1- compute_accuracy(net, trainloader, 1000))* 100,2)
                    train_errors[lr].append(train_error)
                    print(f'Train_error: {train_error}')

                    # compute accuracy on a random sample of 1000 test images
                    test_error = round((1- compute_accuracy(net, testloader, 1000))*100,2)
                    test_errors[lr].append(test_error)
                    print(f'Test_error: {test_error}')

        print('Finished Training')

        #plt.show()
    for i in range(len(learning_rates)): 
        lr = learning_rates[i]
        #plt.figure(figsize=(12, 4))
        # Training Loss Plot
        plt.subplot(1,3,i+1)
        plt.plot(losses[lr])
        plt.title(f'Training Loss (lr={lr})')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
    plt.show()

    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        # Training Error Plot
        plt.subplot(1,3,i+1)
        plt.plot(train_errors[lr])
        plt.title(f'Training Error (lr={lr})')
        plt.xlabel('Iteration')
        plt.ylabel('Training Error %')
        ax = plt.gca()
        # For y-axis
        formatter = ticker.FuncFormatter(lambda y, pos: f'{y:.1f}')
        ax.yaxis.set_major_formatter(formatter)
    plt.show()
    
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        # Test Error Plot
        plt.subplot(1,3,i+1)
        plt.plot(test_errors[lr])
        plt.title(f'Test Error (lr={lr})')
        plt.xlabel('Iteration')
        plt.ylabel('Test Error %')    
        ax = plt.gca()
        # For y-axis
        formatter = ticker.FuncFormatter(lambda y, pos: f'{y:.1f}')
        ax.yaxis.set_major_formatter(formatter)         
    plt.show()


def compute_accuracy(net, dataloader, n_samples):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= n_samples:
                break
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
    
def compute_num_parameters(net:nn.Module):

    num_para = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_para}')
    return num_para



class MobileNetV1(nn.Module):
    """Define MobileNetV1 please keep the strucutre of the class"""
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()
        # Define the layers for MobileNetV1
        def depthwide_conv(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            # Standard convolutional layer
            nn.Conv2d(ch_in, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output size: 32 x 112 x 112 
            
            # 1. Depthwise convolutional layer
            depthwide_conv(32, 64, 1),
            # Output size: 64 x 112 x 112
            # 2. Depthwise convolutional layer
            depthwide_conv(64, 128, 2),
            # Output size: 128 x 56 x 56
            # 3. Depthwise convolutional layer
            depthwide_conv(128, 128, 1),
            # Output size: 128 x 56 x 56
            # 4. Depthwise convolutional layer
            depthwide_conv(128, 256, 2),
            # Output size: 256 x 28 x 28
            # 5. Depthwise convolutional layer
            depthwide_conv(256, 256, 1),
            # Output size: 256 x 28 x 28
            # 6. Depthwise convolutional layer
            depthwide_conv(256, 512, 2),
            # Output size: 512 x 14 x 14
            # 7-12. Depthwise convolutional layer
            depthwide_conv(512, 512, 1),
            depthwide_conv(512, 512, 1),
            depthwide_conv(512, 512, 1),
            depthwide_conv(512, 512, 1),
            depthwide_conv(512, 512, 1),
            # Output size: 512 x 14 x 14
            # 13. Depthwise convolutional layer
            depthwide_conv(512, 1024, 2),
            # Output size: 1024 x 7 x 7
            # 14. Depthwise convolutional layer
            depthwide_conv(1024, 1024, 1),
            # Output size: 1024 x 7 x 7
            # 15. Average pooling layer
            nn.AdaptiveAvgPool2d(1)
        )
        # Define the last fully connected layer
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    
if __name__ == '__main__':
    #train_classifier()
    q1= False
    #Q1
    weight_visualization_ab(q1)
    #Q2
    hypaparameter_sweep()
    # #Q3
    from torchvision import models
    resnet34 = models.resnet34(pretrained=True)
    num_para = compute_num_parameters(resnet34)
    #print(num_para)
    # Q4
    ch_in=3
    n_classes=1000
    model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
    result = model(torch.randn(1, 3, 224, 224))
    print(f'Shape of result after testing on one image: {result.shape}')
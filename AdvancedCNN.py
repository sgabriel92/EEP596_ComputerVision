import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
from scipy import signal

def CIFAR10_dataset_a():

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10/', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    return images, labels

class GAPNet(nn.Module):

    def __init__(self):
        super(GAPNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.gap = nn.AvgPool2d(kernel_size=10,stride=10,padding=0)
        self.fc = nn.Linear(10, 10,bias=True)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.gap(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x 

def train_GAPNet():

    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10/train', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10/test', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net = GAPNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):  # loop over the dataset for 2 iteration
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
    PATH = './cifar10/Gap_net_10epoch.pth'
    torch.save(net.state_dict(), PATH)

def eval_GAPNet():

        # Initialized the network and load from the saved weights
    PATH = './cifar10/Gap_net_10epoch.pth'
    net = GAPNet()
    net.eval()
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

def backbone():

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    model.eval()

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(256,antialias=True),
                                    transforms.CenterCrop(224),                                  
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

    
    image = cv2.imread('cat_eye.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor_img = transform(image)
    tensor_img = tensor_img.unsqueeze(0)
    features = model(tensor_img)
    #print(features.shape)
    #print(features)
    return features     

def transfer_learning():    

    start = time.time()
    #Define cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    #Change architecture of the model
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, 10)
    print(model)
    
    #Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    #Unfreeze the last fully-connected layer``
    for param in model.fc.parameters():
        param.requires_grad = True

    
    #Move model to GPU
    model = model.to(device)

    model.train()
    #Get Data
    batch_size = 4

    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(256,antialias=True),
                                    transforms.CenterCrop(224),                                  
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10/train', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10/test', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):  # loop over the dataset for 10 iteration
        #pass
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs,labels = inputs.to(device),labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
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
    PATH = './Res_net_10epoch.pth'
    torch.save(model.state_dict(), PATH)
    
    model.eval()
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # Evaluates samples
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    end = time.time()
    print("The time of execution of above program is :",((end-start) * 10**3)/60000, "min")


def constructParaboloid(w=256,h=256):
    img = np.zeros((w,h), np.float32)
    for x in range (w):
       for y in range (h):
            # let's center the paraboloid in the img
            img[y,x] = (x-w/2)**2 + (y-h/2)**2
    return img


def newtonMethod(x0, y0, epoch):

    lr = 0.1
    current_point = np.array([x0, y0], dtype=np.float32)

    # Img from paraboloid function
    img = constructParaboloid()

    # Define the kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # image derivatives
    img_dx = cv2.filter2D(img, -1, sobel_x)
    img_dy = cv2.filter2D(img, -1, sobel_y)


    # second-order derivative kernels
    sobel_xx = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]], dtype=np.float32)
    sobel_yy = np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]], dtype=np.float32)
    sobel_xy = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32)

    # second-order image derivatives
    img_dxx = cv2.filter2D(img, -1, sobel_xx)
    img_dyy = cv2.filter2D(img, -1, sobel_yy)
    img_dxy = cv2.filter2D(img, -1, sobel_xy)
    
    def gradient(x, y):
        # Use the image derivatives instead of the analytical derivatives
        return np.array([img_dx[int(y), int(x)], img_dy[int(y), int(x)]])

    def hessian(x, y):
        # Use the second-order image derivatives to construct the Hessian matrix
        return np.array([[img_dxx[int(y), int(x)], img_dxy[int(y), int(x)]], 
                        [img_dxy[int(y), int(x)], img_dyy[int(y), int(x)]]])
    
    for _ in range(epoch):
        grad = gradient(*current_point)
        hess = hessian(*current_point)
        current_point -= lr *np.linalg.inv(hess) @ grad

    print(f'Ending pixel location: {current_point}')
    final_x, final_y = current_point[0], current_point[1]
    #print(f'Ending pixel value: {img[int(final_y), int(final_x)]}')

    return final_x, final_y


if __name__ == "__main__":

    #images, labels = CIFAR10_dataset_a()
    #train_GAPNet()
    #eval_GAPNet()
    #backbone()
    #transfer_learning()
    newtonMethod(50, 70, 50) 

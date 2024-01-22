import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are 
    a=0.37, b=1.37, and c=0.73.  
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    # Given values
    w = torch.tensor([2.0, -3.0, -3.0], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([-1.0, -2.0,1], dtype=torch.float32, requires_grad=True)

    z =  torch.dot(w[:2],x[:2])+w[2]

    a = torch.exp(-(z))
    b = (1+ a)
    c = (1/b)

    a = round(a.item(), 4)
    b = round(b.item(), 4)
    c = round(c.item(), 4)

    #print(f'a={a}')
    #print(f'b={b}')
    #print(f'c={c}')

    return a, b, c
  
def chain_rule_b():
    """
    In the lecture notes, the backward pass values are
    ±0.20, ±0.39, -0.59, and -0.53.  
    Calculate these numbers to 4 decimal digits 
    and return in order of gradients for w0, x0, w1, x1, w2.
    """
     # Given values
    w = torch.tensor([2.0, -3.0, -3.0], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([-1.0, -2.0], dtype=torch.float32, requires_grad=True)

    z = 1 / (1 + torch.exp(-(torch.dot(w[:2], x[:2]) + w[2])))

    z.backward()

    #Get Gradients
    gw0,gw1,gw2 = w.grad
    gx0,gx1 = x.grad
    # Round the gradients to 4 decimal digits
    gw0 = round(gw0.item(), 4)
    gw1 = round(gw1.item(), 4)
    gw2 = round(gw2.item(), 4)
    gx0 = round(gx0.item(), 4)
    gx1 = round(gx1.item(), 4)
    #print(f'gw0 ={gw0}')
    #print(f'gx0 = {gx0}')
    #print(f'gw1 = {gw1}')
    #print(f'gx1 = {gx1}')
    #print(f'gw2 = {gw2}')

    return gw0, gx0, gw1, gx1, gw2

def backprop_a():
    """
    Let f(w,x) = torch.tanh(w0x0+w1x1+w2).  
    Assume the weight vector is w = [w0=5, w1=2], 
    the input vector is  x = [x0=-1,x1= 4],, and the bias is  w2  =-2.
    Use PyTorch to calculate the forward pass of the network, return y_hat = f(w,x).
    """
    #Forwardpass
    w = torch.tensor([5, 2, -2], dtype=torch.float32,requires_grad=True )
    x = torch.tensor([-1, 4], dtype=torch.float32,requires_grad=True)

    y_hat = torch.tanh(torch.dot(w[:2],x[:2]) + w[2])

    #print(f'y_hat = {y_hat}')

    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients 
    for each of the weights, and return the gradient of them 
    in order of w0, w1, and w2.
    """
    #Forwardpass
    w = torch.tensor([5, 2, -2], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([-1, 4], dtype=torch.float32, requires_grad=True)

    y_hat = torch.tanh(torch.dot(w[:2],x[:2]) + w[2])
    # Define the ground truth
    ground_truth = 1.0
    ground_truth = torch.tensor(ground_truth,dtype=torch.float32)

    # Define the MSE loss
    criterion = nn.MSELoss()
    loss = criterion(y_hat, ground_truth)

    #loss = (y_hat - ground_truth) ** 2
    
    loss.backward()
    gw0,gw1,gw2 = w.grad

    #print(f'gw0 ={gw0}')
    #print(f'gw1 = {gw1}')
    #print(f'gw2 = {gw2}')


    return gw0, gw1, gw2

def backprop_c():
    """
    Assuming a learning rate of 0.1, 
    update each of the weights accordingly. 
    For simplicity, just do one iteration. 
    And return the updated weights in the order of w0, w1, and w2 
    """
    #Forwardpass
    w = torch.tensor([5, 2, -2], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([-1, 4], dtype=torch.float32, requires_grad=True)

    y_hat = torch.tanh(torch.dot(w[:2],x[:2]) + w[2])
    # Define the ground truth
    ground_truth = 1.0
    ground_truth = torch.tensor(ground_truth,dtype=torch.float32)

    # Define the MSE loss
    criterion = nn.MSELoss()
    loss = criterion(y_hat, ground_truth)
    
    loss.backward()
    gw0,gw1,gw2 = w.grad

    #print(f'gw0 ={gw0}')
    #print(f'gw1 = {gw1}')
    #print(f'gw2 = {gw2}')

    learning_rate = 0.1

    # Update the weights using gradient descent
    w0 = w[0] - learning_rate * gw0
    w1 = w[1] - learning_rate * gw1
    w2 = w[2] - learning_rate * gw2

    # Print the updated weights
    #print(f"Updated w0: {w0}")
    #print(f"Updated w1: {w1}")
    #print(f"Updated w2: {w2}")

    return  w0, w1, w2 

def stride():

    """
    Apply a stride convolution with stride 2, 
    using a 3x3 Scharr_x filter ( as mentioned in lecture 2) to "cat_eye.jpg." 
    Load the image in grayscale.  
    Return the convolved image in the torch array (2 dimension) with the type torch.FloatTensor.
    (Remember to pad the image with value 0, 
    such that when performing convolve only without stride, the image size after convolution is the same as the original image size.)
    """
    
    image = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)
    #Kernel
    scharr_x = (1/32)*np.array([[3, 0, -3],
                                [10, 0, -10],
                                [3, 0, -3]])

    #Define Transform
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    #Transform image to tensor
    image_tensor = transform(image)
    #Transform kernel to tensor
    kernel_tensor = transform(scharr_x)

    #Secure datype
    kernel_tensor = kernel_tensor.float()
    #Squeeze kernel to rigth shape
    kernel_tensor = torch.unsqueeze(kernel_tensor,0)

    #Convert Image to 0..255
    image_tensor = image_tensor*255


    #Create Padding
    zero_Pad = nn.ZeroPad2d(1)
    image_tensor = zero_Pad(image_tensor)
    
    #Convolve
    stride = 2
    # Apply convolution
    scharr_out = F.conv2d(image_tensor, kernel_tensor, stride = stride,padding='valid')
    scharr_out = scharr_out.squeeze(0)

    #Secure datatype
    scharr_out = torch.FloatTensor(scharr_out)

    #Save image
    torchvision.utils.save_image(scharr_out,'scharr_out.jpg')

    return scharr_out

def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random. 
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor. 
    The first dimension of the array should be batch dimension, the second channel dimension, 
    followed by image height and image width. 
    2. Labels of the images in a torch array

    """
    #Define Transform
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #Define batch_size
    batch_size = 4
    #Get Datasets and create Dataloader
    trainset = torchvision.datasets.CIFAR10(root='./cifar10/train', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./cifar10/test', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    #Define Classes
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #Visualization of image samples
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        #plt.show()


    #Get random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    #Show images
    imshow(torchvision.utils.make_grid(images))
    plt.axis('off')
    for i in range(4):
        plt.text(i*34, -10, f'{classes[labels[i]]}', color='red',verticalalignment='top', fontsize=12)

    # print labels
    # Show the plot
    #plt.show()

    return images, labels

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


if __name__ == "__main__":
    # your text code here
    #a,b,c = chain_rule_a()
    #gw0, gx0, gw1, gx1, gw2 = chain_rule_b()
    #y_hat = backprop_a()
    #gw0, gw1, gw2  = backprop_b() 
    #w0, w1, w2  = backprop_c() 
    scharr_out = stride()
    #images, labels = CIFAR10_dataset_a()
    #train_classifier()
    #evalNetwork()

import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt

class assignment3():
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, img):
        #print("-------Q1a--------")
        #Convert Image Color Space
        torch_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #Print current specifications
        #print(f'Original OpenCV Image shape: {torch_img.shape}')
        #print(f'Original OpenCV Image dtype: {torch_img.dtype}')
        #print(f'Original OpenCV Image Pixel dtype: {torch_img[0,0,0].dtype}')
        #Transformation to torch tensor
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        torch_img = transform(torch_img)
        #Create correct order
        torch_img = torch_img.permute(1, 2, 0)
        #Print Results
        #print(f'Torch Image shape: {torch_img.size()}')
        #print(f'Torch Image dtype: {torch_img.dtype}')
        #print(f'Torch Image dtype of Pixel {torch_img[0,0,:]} dtype: {torch_img[0,0,0].dtype}')
        
        #plt.imshow(torch_img)
        #plt.axis('off')
        #plt.savefig('Q1aImage.png',bbox_inches='tight')
        #plt.show()
        return torch_img
    
    def brighten(self, torch_img):
        #print("-------Q1b--------")

        #Change range unit8
        bright_img = torch_img*255
        bright_img = bright_img + 100

        #Clamp to min and max values
        bright_img = torch.clamp(bright_img, min=0, max=255, out=None)
        bright_img = bright_img / 255

        #Print results
        #print(f'Torch Image shape: {bright_img.size()}')
        #print(f'Torch Image dtype: {bright_img.dtype}')
        #print(f'Torch Image dtype of Pixel {bright_img[0,0]} dtype: {bright_img[0,0,0].dtype}')

        #plt.imshow(bright_img)
        #plt.axis('off')
        #plt.savefig('Q1bImage.png',bbox_inches='tight')
        #plt.show()
        return bright_img
    
    def saturation_arithmetic(self, img):
        #print("-------Q1c--------")
        
        #Fill your code here
        #Convert color order
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #apply torch transfrom
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        torch_img = transforms(image)
        torch_img = torch_img.permute(1, 2, 0)
        #convert to unit8
        torch_img = torchvision.transforms.functional.convert_image_dtype(torch_img, torch.uint8)
        #print results
        #print(f"Image dimensions: {torch_img.shape}")
        #print(f"Image data type: {torch_img.dtype}")
        #print(f"Pixel data type {torch_img[0,0]}: {torch_img[0,0].dtype}")

        #Ensure that values are at a maxiumum of 255
        saturated_img = torch.where(255 - torch_img < 100, torch.tensor(255),torch_img+100)
        saturated_img = torch.clamp(saturated_img, min=0, max=255, out=None)

        #print results
        #print(f"Saturated Image dimensions: {saturated_img.shape}")
        #print(f"Saturated Image data type: {saturated_img.dtype}")
        #print(f"Saturated Image Pixel data type {saturated_img[0,0]}: {saturated_img[0,0].dtype}")


        #plt.imshow(saturated_img)
        #plt.axis('off')
        #plt.savefig('Q1cImage.png',bbox_inches='tight')
        #plt.show()

        return saturated_img
    
    def add_noise(self, torch_img):
        #print("-------Q2--------")

        #Assign variables
        stdDiv = 100.0
        mean =0
        height,width,cha = torch_img.shape
        #create normal distribution
        noise = np.random.normal(mean, stdDiv, (height, width, 1))
        numpy_as_tensor = torch.tensor(noise)
        print(f"Data Type {torch_img.dtype}")
        #print(numpy_as_tensor[0,0])
        #Convert to range between 0-255
        torch_img = torch_img *255
        #Do we need conversion?
        #Add noise
        torch_img = torch_img+numpy_as_tensor
        #torch_img = torch_img+noise
        #print(f"Data Type {torch_img.dtype}")
        #print(f"Data Type {torch_img[0,0]}")
        
        #Clamp to values between 0-255
        noisy_img = torch.clamp(torch_img, 0,255,out=None) 
        noisy_img = noisy_img/ 255
        #Transform datatype
        noisy_img = torchvision.transforms.functional.convert_image_dtype(noisy_img, torch.float32)

        #plt.imshow(noisy_img)
        #plt.savefig('Q2Image.png',bbox_inches='tight')
        #plt.show()

        return noisy_img
    
    def normalization_image(self, img):
        #print("-------Q3a--------")
        #Fill your code hear
        #Convert color order
        torch_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #Transform to tensor,change order and assign correct data type
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        torch_img = transform(torch_img)
        torch_img = torch_img.permute(1, 2, 0)
        torch_img = torchvision.transforms.functional.convert_image_dtype(torch_img, torch.float64)

        mean = torch.mean(torch_img,dim=(0, 1))
        std = torch.std(torch_img,dim=(0, 1))
    
        #print(f'Mean before normalization: {mean}')
        #print(f'Std before normalization: {std}')

        torch_img = torch_img - mean
        image_norm = torch_img / std
        
        mean2 = torch.mean(image_norm,dim=(0, 1))
        std2 = torch.std(image_norm,dim=(0, 1))
        
        #print(f'Mean after normalization: {torch.round(mean2,decimals=5)}')
        #print(f'Std after normalization: {torch.round(std2,decimals=5)}')
        #print(f"Data Type {image_norm.dtype}")
        #print(f"Data Type {image_norm[0,0]}")
        image_norm = torch.clamp(image_norm, 0, 1)

        #plt.imshow(image_norm)
        #plt.savefig('Q3aImage.png',bbox_inches='tight')
        #plt.show()

        return image_norm
    
    def Imagenet_norm(self, img):
        #print("-------Q3b--------")
        #Fill your code hear
        #Convert image
        torch_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #Performa transformation
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        torch_img = transform(torch_img)
        torch_img = torch_img.permute(1, 2, 0)
        torch_img = torchvision.transforms.functional.convert_image_dtype(torch_img, torch.float64)

        # Define the ImageNet means and standard deviations for each channel
        imagenet_means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64)
        imagenet_stds = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)
        imagenet_means = imagenet_means[None,None,:]
        imagenet_stds = imagenet_stds[None,None,:]

        torch_img = torch_img - imagenet_means
        ImageNet_norm = torch_img / imagenet_stds

        mean2 = torch.mean(ImageNet_norm,dim=(0, 1))
        std2 = torch.std(ImageNet_norm,dim=(0, 1))

        #print(f'Mean after normalization: {mean2}')
        #print(f'Std after normalization: {std2}')

        ImageNet_norm = torch.clamp(ImageNet_norm, 0, 1)


        #plt.imshow(ImageNet_norm)
        #plt.savefig('Q3bImage.png',bbox_inches='tight')
        #plt.show()
        return ImageNet_norm

    
    def dimension_rearrange(self, img):
        print("-------Q4--------")
        #Fill your code hear
        #Convert image
        rearrange = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #Apply transformation
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        rearrange = transform(rearrange)
        rearrange = torch.unsqueeze(rearrange,0)
        print(rearrange.dtype)
        print(rearrange.shape)
        
        
        return rearrange
    
    def chain_rule(self,x,y,z):
        print("-------Q5--------")
        #Fill your code hear
        f = x*y+z
        q = x*y

        df_dx = float(y)
        df_dy = float(x)
        df_dz = float(1)
        df_dq = float(1)

        print(f'df/dx={df_dx}, df/dy = {df_dy}, df/dz = {df_dz}, df/dq = {df_dq}')

        return df_dx, df_dy, df_dz, df_dq
    
    def relu(self,x,w):
        #print("-------Q6--------")
        #Fill your code hear
        x0 = x[0]
        x1 =x[1]
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]

        #Forward pass
        dot = x0*w0 + x1*w1 + w2
        #print(f"dot = {dot}")

        #ReLU activation
        f = max(0.0,dot)
        #print(f"f = {f}")

        ddot = 1 if f > 0 else 0

        print(ddot)

        dx = np.float32([w[0]*ddot,w[1]*ddot])
        dw = np.float32([x[0]*ddot,x[1]*ddot,1.0*ddot])
       
        #print(f"dx = {dx}")
       # print(f"dw = {dw}")

        #print(type(dx[0]))
        #print(type(dx))
        

        return dx, dw
    


img = cv.imread('original_image.PNG')
assign = assignment3()
torch_img = assign.torch_image_conversion(img)
bright_img = assign.brighten(torch_img)
saturated_img = assign.saturation_arithmetic(img)
noisy_img = assign.add_noise(torch_img)
image_norm =  assign.normalization_image(img)
ImageNet_norm =  assign.Imagenet_norm(img)
rearrange =  assign.dimension_rearrange(img)
df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2,y=5,z=-4)
dx, dw = assign.relu(x=[-1,2],w=[2,-3,-3])
# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

    # Define the fill color (e.g., bright green)
    fill_color = (0, 255, 0)
    # Create a copy of the input image to keep the original image unchanged
    output_image = self.ant_img.copy()
  
    #Seed coordinates
    seed_y = seed[0]
    seed_x = seed[1]
    
    #Image Dimensions
    height,width = output_image.shape[:2]
    
    #Logic Check: Is seed pixel within boundarys
    if (0 <= seed_x < width) and (0 <= seed_y < height):
      #Get original color
      ori_color = output_image[seed_y][seed_x].copy()
      
      #Define coordinate offset for neighboring pixels
      offsets = [(-1, 0),(0,-1),(0, 1),(1, 0)]

      # Define a stack for floodfill
      stack = []

      #Floodfill algorithm
      if (ori_color != fill_color).any():
        #Append seed pixel to stack
        stack.append(seed)
        #Asign fill color to seed pixel
        output_image[seed_y][seed_x] = fill_color
        #While stack is not empty, keep assigning the fill color to the pixel
        while len(stack) > 0:
          #Take LIFO pixel from stack
          pixel = stack.pop()
          #Empty aarry of neighbor pixels
          neighbors=[] 
          #Define neighbors of selected pixel
          for dy, dx in offsets: 
            new_x = pixel[1] + dx
            new_y = pixel[0] + dy
            # Check if the new coordinates are within the image boundaries
            if 0 <= new_x < width and 0 <= new_y < height:
              neighbors.append((new_y, new_x))
          
          #Loop through neighbors and assign fill color
          for q in neighbors:
            if (output_image[q[0]][q[1]] == ori_color).all():
              stack.append(q)
              output_image[q[0]][q[1]] = fill_color
    #else:
      #print("Seed coordinates are out of bounds. Choose a different seed pixel.")

    cv2.imwrite('floodfill.jpg', output_image)
    return output_image

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    #kernel = # 1D Gaussian kernel
    image = self.cat_eye.copy()
    #Dimensions of Image
    height, width = image.shape

    #Create padded image
    padded_image = np.zeros((height+2, width+2), dtype = float)
    for y in range(height):
      for x in range(width):
        padded_image[y+1][x+1] = image[y][x]

    #Dimensions of padded image
    height_padded, width_padded = padded_image.shape

    #Create List of blurred Images
    self.blurred_images = []

    #Kernel
    k = np.array([0.25,0.5,0.25])
    w = len(k)
    mu = w//2

    #Create temp images for correct image processing
    temp = np.zeros_like(padded_image)
    out = padded_image

    for i in range(5):
      #Track iteration accordingly
      #Create new result image
      result = np.zeros_like(image)
      #Reset temp image
      temp = np.zeros_like(padded_image)
      #assign current image value of previous
      current_image = out

      # Apply convolution
      #Horizontal convolution
      for y in range(height_padded):
        for x in range(mu,width_padded-mu):
          val = 0
          for j in range(w):
            val += k[j]*current_image[y][x+mu-j]
          temp[y][x] = val
      
      #Vertical Convolution
      for y in range(mu,height_padded-mu):
        for x in range(width_padded):
          val = 0
          for j in range(w):
            val += k[j]*temp[y+mu-j][x]
          out[y][x] = val

      #Cut of padding and store the blurred image
      for y in range(height_padded-2):
        for x in range(width_padded-2):
          result[y][x]=out[y+1][x+1]
      
      #print(result.dtype)
      self.blurred_images.append(result)       
      cv2.imwrite(f'gaussain blur {i}.jpg', result)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define the first 1D kernel
    vertical_kernel = np.array([0.25, 0.5, 0.25])
    
    # Define the second 1D kernel
    horizontal_kernel = np.array([0.5, 0, -0.5])

    # Flip the kernels manually
    vertical_kernel_flipped = vertical_kernel[::-1]
    horizontal_kernel_flipped = horizontal_kernel[::-1]
    
    # Calculate the length of each kernel
    vertical_length = len(vertical_kernel_flipped)
    horizontal_length = len(horizontal_kernel_flipped)

    # Initialize a 2D matrix with zeros
    filter_kernel = np.zeros((horizontal_length, vertical_length))

    # Calculate the outer product manually
    for i in range(horizontal_length):
        for j in range(vertical_length):
            filter_kernel[i, j] = horizontal_kernel_flipped[i] * vertical_kernel_flipped[j]

    #print(filter_kernel)
    kernel_height,kernel_width = filter_kernel.shape
    mu_y = kernel_height // 2
    mu_x = kernel_width // 2
    
    # Store images
    self.vDerive_images = []
    
    for g in range(5):
      # Apply horizontal and vertical convolution
      image = self.blurred_images[g].copy()

      #Create padded image
      #Extract height and width of image
      height, width = image.shape
      padded_image = np.zeros((height+2, width+2),dtype=image.dtype)
      for y in range(height):
        for x in range(width):
          padded_image[y+1][x+1] = image[y][x]
      #Extract height and width of padded image
      height_padded, width_padded = padded_image.shape

      #Initialize a temp image to perform convolution
      temp = np.zeros_like(padded_image,dtype=float)
      out = np.zeros_like(image,dtype=float)

      for i in range(mu_y,height_padded-mu_y):
        for j in range(mu_x,width_padded-mu_x):
            val = 0
            for y in range(kernel_height):
                for x in range(kernel_width):
                    # Calculate the corresponding position in the image
                    image_y = i + y - mu_y
                    image_x = j + x - mu_x
                    val += padded_image[image_y, image_x] * filter_kernel[y, x]
                    temp[i, j] = val
      
      #Cut off padding
      for y in range(height_padded-2*mu_y):
        for x in range(width_padded-2*mu_x):
          out[y][x]=temp[y+mu_y][x+mu_x]
      
      out = np.clip(out,0,255,out).astype(np.uint8)
      #print(out.dtype)
      self.vDerive_images.append(out)
      cv2.imwrite(f'vertical {g}.jpg', out)
    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
      #Define the first 1D 
      vertical_kernel = np.array([0.5, 0, -0.5])

      # Define the second 1D 
      horizontal_kernel = np.array([0.25, 0.5, 0.25])

      # Flip the kernels manually
      vertical_kernel_flipped = vertical_kernel[::-1]
      horizontal_kernel_flipped = horizontal_kernel[::-1]

      # Calculate the length of each kernel
      vertical_length = len(vertical_kernel_flipped)
      horizontal_length = len(horizontal_kernel_flipped)
      

      # Initialize a 2D matrix with zeros
      filter_kernel = np.zeros((horizontal_length, vertical_length))

      # Calculate the outer product manually
      for i in range(horizontal_length):
          for j in range(vertical_length):
              filter_kernel[i, j] = horizontal_kernel_flipped[i] * vertical_kernel_flipped[j]
      #print(filter_kernel)
      kernel_height,kernel_width = filter_kernel.shape
      mu_y = kernel_height // 2
      mu_x = kernel_width // 2

      # Store images
      self.hDerive_images = []
      
      for g in range(5):
        # Apply horizontal and vertical convolution
        image = self.blurred_images[g].copy()
        
        #Create padded image
        #Extract height and width of image
        height, width = image.shape
        padded_image = np.zeros((height+2, width+2),dtype=image.dtype)
        for y in range(height):
          for x in range(width):
            padded_image[y+1][x+1] = image[y][x]
        
        #Extract height and width of padded image
        height_padded, width_padded = padded_image.shape

        #Initialize a temp image to perform convolution
        temp = np.zeros_like(padded_image,dtype=float)
        out = np.zeros_like(image,dtype=float)

        for i in range(mu_y,height_padded-mu_y):
          for j in range(mu_x,width_padded-mu_x):
              val = 0
              for y in range(kernel_height):
                  for x in range(kernel_width):
                      # Calculate the corresponding position in the image
                      image_y = i + y - mu_y
                      image_x = j + x - mu_x
                      val += padded_image[image_y, image_x] * filter_kernel[y, x]
                      temp[i, j] = val
        
        #Cut off padding
        for y in range(height_padded-2*mu_y):
          for x in range(width_padded-2*mu_x):
            out[y][x]=temp[y+mu_y][x+mu_x]
        
        out = np.clip(out,0,255,out).astype(np.uint8)
        #print(out.dtype)
        self.hDerive_images.append(out)
        cv2.imwrite(f'horizontal {g}.jpg', out)
      return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]
    
    for i, (vimg, himg) in enumerate(zip(self.vDerive_images, self.hDerive_images)):

      out = np.sqrt(vimg**2 + himg**2)

      out = np.clip(out,0,255,out).astype(np.uint8)
      #print(out.dtype)
      
      self.gdMagnitute_images.append(out)
      #cv2.imwrite(f'gradient {i}.jpg', out)
    return self.gdMagnitute_images
    
  def scipy_convolve(self):
    image = self.cat_eye.copy()
    #Kernel
    kernel = np.array([0.25, 0.5, 0.25])

    #Store outputs
    self.scipy_smooth = []

    #Create temp images for correct image processing
    out = image
    for i in range(5):
      #Track iteration accordingly
      #assign current image value of previous
      current_image = out

      out = np.zeros_like(image)
      # Apply horizontal convolution
      out = scipy.signal.convolve2d(current_image, kernel.reshape(1, -1), mode='same', boundary='fill', fillvalue=0)

      # Apply vertical convolution
      out = scipy.signal.convolve2d(out, kernel.reshape(-1, 1), mode='same', boundary='fill', fillvalue=0)

      out = np.clip(out,0,255,out).astype(np.uint8)
      #print(out.dtype)
      self.scipy_smooth.append(out)

      cv2.imwrite(f'scipy smooth {i}.jpg', out)

    for i ,(imblurr, imscip) in enumerate(zip(self.blurred_images, self.scipy_smooth)):
      #print(f'Iteration {i}')
      difference = np.abs(imblurr - imscip)
      max_diff = np.max(difference)
      print(max_diff)
      # Check if the maximum difference is within 5
      if max_diff <= 5:
        print("The maximum difference is within 5.")
      else:
        print("The maximum difference is greater than 5.")
    
    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]
    for _ in range(num_repetitions):
      # Perform 1D conlve
      out = np.convolve(out,box_filter,mode='full')

    # Print the resulting filter
    print(f"Convolved Filter: {out}")      

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    floodfill_img = ass.floodfill((100, 100))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def scanlines(tb_left, tb_right):
    disparities = np.zeros(100)

    tb_left = tb_left[152,102:202]
    tb_right = tb_right[152,102:202]


    
    window_size = 3

    for i in range(window_size//2, 100-window_size//2):
        best_match = float('inf')
        best_match_index = -1
        left_window = tb_left[i-window_size//2:i+window_size//2+1]
        for j in range(window_size//2, 100-window_size//2):
            right_window = tb_right[j-window_size//2:j+window_size//2+1]
            ssd = np.sum((left_window - right_window)**2)
            if ssd < best_match:
                best_match = ssd
                best_match_index = j

        disparities[i] = i - best_match_index

    max_disparity = max(disparities)

    #_______________________________________________________

    return max_disparity

def auto_correlation(tb_right):
    path = 'Images/auto_correlation/'

    n = 30
    auto_correlation_values = []

    for i in range(n+1):
        tb_right_shifted = np.zeros_like(tb_right)
        tb_right_shifted[:,i:] = tb_right[:,:tb_right.shape[1]-i]
        abs_difference = np.abs(tb_right_shifted -tb_right)
        auto_correlation_values.append(abs_difference[152,152])
        #cv.imwrite(os.path.join(path , f'abs_difference_{i}.png'), abs_difference)

    # print(f'Auto-correlation values: {auto_correlation_values}')

    # Visualization
    # plt.figure()
    # plt.title('Auto Correlation')
    # plt.xlabel('Shift')
    # plt.ylabel('Absolute Difference')
    # plt.plot(auto_correlation_values)
    # plt.show()

    return auto_correlation_values

def smoothing(tb_right):
    path = 'Images/smoothing/'

    n = 30
    smoothened_corr_values = []

    for i in range(n+1):
        tb_right_shifted = np.zeros_like(tb_right)
        tb_right_shifted[:,i:] = tb_right[:,:tb_right.shape[1]-i]
        abs_difference = np.abs(tb_right_shifted-tb_right)
        abs_difference = cv.boxFilter(abs_difference, -1, (5,5))
        smoothened_corr_values.append(abs_difference[152,152])
        #cv.imwrite(os.path.join(path,f'abs_diff_smooth_{i}.png'), abs_difference)
    
    # print(f'Smoothened correlation values: {smoothened_corr_values}')

    # Visualization
    # plt.figure()
    # plt.title('Auto Correlation with Smoothing')
    # plt.xlabel('Shift')
    # plt.ylabel('Absolute Difference')
    # plt.plot(smoothened_corr_values)
    # plt.show()

    return smoothened_corr_values

def cross_correlation(tb_left, tb_right):
    path = 'Images/cross_correlation/'
    n = 30
    cross_correlation_values = []

    for i in range(n+1):
        tb_right_shifted = np.zeros_like(tb_right)
        tb_right_shifted[:,i:] = tb_right[:,:tb_right.shape[1]-i]

        abs_difference = np.abs(tb_right_shifted - tb_left)
        abs_difference = cv.boxFilter(abs_difference, -1, (5,5))
        cross_correlation_values.append(abs_difference[152,152])
        #cv.imwrite(os.path.join(path, f'abs_cross_corr_{i}.png'), abs_difference)
        
    # print(f'Cross correlation values: {cross_correlation_values}')

    # Visualization
    # plt.figure()
    # plt.title('Auto Cross Correlation ')
    # plt.xlabel('Shift')
    # plt.ylabel('Absolute Difference')
    # plt.plot(cross_correlation_values)
    # plt.show()

    return cross_correlation_values

def disparity_map(tb_left, tb_right):
    n = 30
    
    disparity_map_output = np.zeros_like(tb_left)

    tensor_list = []

    for i in range(n+1):
        # shift and zero padding
        tb_right_shifted = np.zeros_like(tb_right)
        tb_right_shifted[:,i:] = tb_right[:, :tb_right.shape[1]-i]
        abs_difference = np.abs(tb_right_shifted-tb_left)
        abs_difference = cv.boxFilter(abs_difference, -1, (5,5))
        tensor_list.append(abs_difference)

    for y in range(tb_left.shape[0]):
        for x in range(tb_left.shape[1]):
            disparity = 0
            cross_correlation_values = []
            for i in range(n+1):
                cross_correlation_values.append(tensor_list[i][y,x])

            disparity = np.argmin(cross_correlation_values)
            disparity_map_output[y,x] = disparity

    # # cv.imwrite('disparity_map.png', disparity_map_output)
    plt.imshow(disparity_map_output, cmap='gray')
    plt.show()

    return disparity_map_output

def right_left_disparity(tb_left, tb_right):
    n = 30  
    right_left_disparity_output = np.zeros_like(tb_right)

    tensor_list = []

    for i in range(n+1):
        # shift and zero padding
        tb_left_shifted = np.zeros_like(tb_left)
        tb_left_shifted[:,:-i or None] = tb_left[:, i:]
        abs_difference = np.abs(tb_left_shifted-tb_right)
        abs_difference = cv.boxFilter(abs_difference, -1, (5,5))
        tensor_list.append(abs_difference)

    for y in range(tb_right.shape[0]):
        for x in range(tb_right.shape[1]):
            disparity = 0
            cross_correlation_values = []
            for i in range(n+1):
                cross_correlation_values.append(tensor_list[i][y,x])

            disparity = np.argmin(cross_correlation_values)
            right_left_disparity_output[y,x] = disparity

    # cv.imwrite('right_left_disparity_output.png', right_left_disparity_output)
    plt.imshow(right_left_disparity_output, cmap='gray')
    plt.show() 
    return right_left_disparity_output

def disparity_check(tb_left, tb_right):
    disparity_check_output = np.zeros_like(tb_left)
    left_right_disparity_map = disparity_map(tb_left, tb_right)
    right_left_disparity_map = right_left_disparity(tb_left, tb_right)

    for y in range(left_right_disparity_map.shape[0]):
        for x in range(left_right_disparity_map.shape[1]):
            d = left_right_disparity_map[y, x]

            if x - d >= 0 and d-5 <= right_left_disparity_map[y, x - d] <= d+5: # check if the right to left disparity is within the range +-5
                disparity_check_output[y, x] = d

    # cv.imwrite('disparity_check_output.png', disparity_check_output)
    plt.imshow(disparity_check_output, cmap='gray')
    plt.show() 
    return disparity_check_output

def reconstruction(tb_left, tb_right):
     img_bw_left = cv.cvtColor(tb_left, cv.COLOR_BGR2GRAY)
     img_bw_right = cv.cvtColor(tb_right, cv.COLOR_BGR2GRAY)
     disparity_map = disparity_check(img_bw_left, img_bw_right)

     focal_length = tb_left.shape[1]
     baseline = 1.0

     points = []
     for y in range(disparity_map.shape[0]):
        for x in range(disparity_map.shape[1]):
            d = disparity_map[y, x]
            if d > 0:  # ignore zero disparities
                Z = focal_length * baseline / d
                X = x
                Y = y
                B, G, R = tb_left[y, x]
                points.append((X, Y, Z, R, G, B))

     with open('kermit.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(points)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        for point in points:
            f.write('{} {} {} {} {} {}\n'.format(*point))
    
    #Visulize the 3D point cloud
    #  infile = 'kermit.ply'    
    #  fig = plt.figure(figsize=(10,10))
    #  ax = fig.add_subplot(111, projection='3d')
    #  data = np.loadtxt(infile, delimiter=' ',skiprows=10)
    #  x,y,z = data[:,0], data[:,1], data[:,2]
    #  r,g,b = data[:,3]/255, data[:,4]/255, data[:,5]/255
    #  # Normalize the RGB values and create an array of colors
    #  colors = np.array([r,g,b]).T

    #  ax.scatter(x, y, z, c=colors, marker='o')
    #  ax.set_xlabel('X Label')
    #  ax.set_ylabel('Y Label')
    #  ax.set_zlabel('Z Label')
    #  plt.show()

if __name__ == "__main__":
    """
    Insert your code here
    """
    # Q1
    image_left = cv.imread('tsukuba_left.png',cv.IMREAD_COLOR)
    image_right = cv.imread('tsukuba_right.png',cv.IMREAD_COLOR)

    gray_image_left = cv.imread('tsukuba_left.png',cv.IMREAD_GRAYSCALE)
    gray_image_right = cv.imread('tsukuba_right.png',cv.IMREAD_GRAYSCALE)

    # Q2
    max_disparity = scanlines(gray_image_left, gray_image_right)
    print("Maximum disparity: ", max_disparity)

    # Q3
    auto_correlation = auto_correlation(gray_image_right)

    # # Q4
    smoothened_corr_values = smoothing(gray_image_right)

     # Q5
    cross_correlation_values = cross_correlation(gray_image_left, gray_image_right)

    # Q6
    disparity_output = disparity_map(gray_image_left, gray_image_right)

    # Q7
    right_left_disparity_output = right_left_disparity(gray_image_left, gray_image_right)

    # Q8
    disparity_check_output = disparity_check(gray_image_left, gray_image_right)

    # Q9
    reconstruction(image_left, image_right)





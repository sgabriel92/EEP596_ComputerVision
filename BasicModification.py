
import cv2
import numpy as np
import os


class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
        
    def check_package_versions(self):
        #Ungraded
        import numpy as np
        import matplotlib
        import cv2
        
        #print(np.__version__)
        #print(matplotlib.__version__)
        #print(cv2.__version__)
        
    def load_and_analyze_image(self):
        
        """
        Fill your code here
        """
        #Image Data
        Image_data = self.image
        Image_data_type = self.image.dtype
        #Pixel Data
        Pixel_data = self.image[0,0,0]
        Pixel_data_type = Pixel_data.dtype
        #Dimensions
        Image_shape = self.image.shape

        #Results
        #print(f"Image data: {Image_data}")
        #print(f"Image data type: {Image_data_type}")
        #print(f"Pixel data : {Pixel_data}")
        #print(f"Pixel data type: {Pixel_data_type}")
        #print(f"Image dimensions: {Image_shape}")
        
        return Image_data_type, Pixel_data_type, Image_shape
        
    def create_red_image(self):
        """
        Fill your code here
        """
        #Create copy of orginal file
        red_image = self.image.copy()
        #Split image into three color channels blue,gree, red                    
        b,g,r = cv2.split(red_image)
        #Create a np.array with zeros and the same shape as input np.array b of blue values                    
        zeros = np.zeros_like(b)
        #Merge all channels together                        
        red_image = cv2.merge((zeros, zeros, r))
        #Save to folder         
        #cv2.imwrite('red_image.PNG',red_image)
        return red_image

    def create_photographic_negative(self):
        """
        Fill your code here
        """
        #Create copy of orginal file
        negative_image = self.image.copy()
        #Calculate negativ values for all channels and all pixels
        negative_image = 255-negative_image

        #Save to folder
        #cv2.imwrite('negative_image.PNG',negative_image)
        return negative_image

    def swap_color_channels(self):
        """
        Fill your code here
        """
        #Create copy of orginal file
        swapped_image = self.image.copy()
        #Split image into three color channels blue,gree, red 
        b,g,r = cv2.split(swapped_image)
        #Merge it back together with swapped blue and red channels
        swapped_image = cv2.merge((r, g, b))
        #Save to folder
        #cv2.imwrite('swapped_image.PNG',swapped_image)
        return swapped_image
    
    def foliage_detection(self):
        """
        Fill your code here
        """
        #Create copy of orginal file
        foliage_image = self.image.copy()
        #Split image into three color channels blue,gree, red 
        b,g,r = cv2.split(foliage_image)
        #Get dimensions of new np.arrays
        row,col = g.shape
        #Iterate through np.array and check of filter conditions. Replace value in np.array of green channel
        for x in range(row):
            for y in range(col):
                if g[x][y] >= 50 and  b[x][y]<50 and r[x][y]<50:
                    g[x][y] =255
                else:
                    g[x][y] =0
        foliage_image = g
        #Save to folder
        #cv2.imwrite('foliage_image.PNG',foliage_image)
        return foliage_image

    def shift_image(self):
        """
        Fill your code here
        """
        #Create copy of orginal file
        shifted_image = self.image.copy()
        #Define shift in X and Y direction
        shiftX = 200
        shiftY = 100
        #Create Translation Matrix
        M = np.float32([[1, 0, shiftX],
	                    [0, 1, shiftY]])
        #Apply translation to image
        shifted_image = cv2.warpAffine(shifted_image, M, (shifted_image.shape[1], shifted_image.shape[0]))
        #Save to folder
        #cv2.imwrite('shifted_image.PNG',shifted_image)
        return shifted_image    

    def rotate_image(self):
        """
        Fill your code here
        """
        #Create copy of orginal file
        rotated_image = self.image.copy()
        #Rotate Image clockwise by 90 degrees 
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
        #Save to folder
        #cv2.imwrite('rotated_image.PNG',rotated_image)
        return rotated_image
        
    def similarity_transform(self, scale, theta, shift):
        """
        Fill your code here
        """ 
        #Create copy of orginal file
        transformed_image = self.image.copy()
        #Get dimensions of image
        rows,cols,chan = transformed_image.shape
        #Define center for rotation matrix
        center = (cols/2,rows/2)
        #Create rotation matrix
        M = cv2.getRotationMatrix2D(center,theta,scale)
        #Add shift for translation
        M[0:,2] += shift
        #Inverse mapping
        M_inv = cv2.invertAffineTransform(M)
        transformed_image = cv2.warpAffine(transformed_image,M_inv,(cols,rows),flags=cv2.INTER_NEAREST)
        #Save to folder
        #cv2.imwrite('transformed_image.PNG',transformed_image)
        return transformed_image
    
    def convert_to_grayscale(self):
        """
        Fill your code here
        """
        #Create copy of orginal file
        original_image = self.image.copy()
        #Split image into three color channels blue,gree, red 
        b,g,r = cv2.split(original_image)
        #Create identical np.array for result
        gray_image = np.zeros_like(b)
        #Get dimensions of np.array
        row,col = gray_image.shape
        #Iterate through all elements calcualte resulting gray value and set value in resulting np.array gray_image
        for x in range(row):
            for y in range(col):
                rl = original_image[x][y][2]
                gl = original_image[x][y][1]
                bl = original_image[x][y][0]
                gray = (3 * rl + 6 * gl + 1 * bl)/10
                gray_image[x][y] = gray

        #Save to folder
        #cv2.imwrite('gray_image.PNG',gray_image)
        return gray_image
    
    def compute_moments(self):
        """
        Fill your code here
        """
        #Create copy of binary image
        binary_image = self.binary_image.copy()
        #Get dimension of binary_image
        rowY,colX = binary_image.shape

        #First-Order Moments
        #Utilize private helper function (__compute_mpq)
        m00 = assignment.__compute_mpq(colX,rowY,0,0,binary_image)
        m10 = assignment.__compute_mpq(colX,rowY,1,0,binary_image)
        m01 = assignment.__compute_mpq(colX,rowY,0,1,binary_image)

        #Centralized Moments
        #Utilize private helper function (__centralizedMoments)
        x_bar = assignment.__centralizedMoments(m10,m00)
        y_bar = assignment.__centralizedMoments(m01,m00)

        #Second-Order Moments
        #Utilize private helper function (__compute_mpq) 
        m11 = assignment.__compute_mpq(colX,rowY,1,1,binary_image)
        m20 = assignment.__compute_mpq(colX,rowY,2,0,binary_image)
        m02 = assignment.__compute_mpq(colX,rowY,0,2,binary_image)

        #Second-Order Centralized
        mu11 = m11 - y_bar*m10  
        mu20 = m20 - x_bar*m10
        mu02 = m02 - y_bar*m01

        # Print the results
        #print("First-Order Moments:")
        #print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        #print("Centralized Moments:")
        #print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        #print("Second-Order Centralized Moments:")
        #print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}") 
        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11
        
    def compute_orientation_and_eccentricity(self):
        """
        Fill your code here
        """
        #Get necessary variables from compute_moments
        mu20, mu02, mu11 = assignment.compute_moments()[5:]

        #Calculate orientation in degrees
        orientation = np.degrees(0.5 *np.arctan2(2*mu11,mu20-mu02))

        #Calculate eccentricity
        eccentricity = np.sqrt((2*np.sqrt((mu20-mu02)**2+4*mu11**2))/(mu20+mu02+np.sqrt((mu20-mu02)**2+4*mu11**2)))

        #print(f"Orientation: {orientation}")
        #print(f"Eccentricity: {eccentricity}")
        return orientation, eccentricity        


    #Helper functions to keep clean code
    def __compute_mpq(self,col,row,p,q,img):
        m = 0
        for x in np.arange(col):
            for y in np.arange(row):
                m += ((x)**p)*((y)**q)*img[y][x]
        return m

    def __centralizedMoments(self,mpq,m00):
        mc = mpq/m00
        return mc


if __name__ == "__main__":
    
    #Instantiate Class and load images
    assignment = ComputerVisionAssignment('original_image.PNG','binary_image.png')

    # Task 0: Check package versions
    assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # Task 2: Create a red image
    red_image = assignment.create_red_image()  
    
    # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()

    # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()    
    
    # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()
    
    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()
    
    # Task 8: Similarity transform  
    transformed_image = assignment.similarity_transform(scale=2.0, theta=45.0, shift=[100, 100])

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()
    
    # Task 10: Moments of a binary image
    assignment.compute_moments()
    
    # Task 11: Orientation and eccentricity of a binary image   
    orientaion, eccentricity =assignment.compute_orientation_and_eccentricity()
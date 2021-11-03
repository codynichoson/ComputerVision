import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
import math
import scipy.signal as sci

class Canny:
    def GaussSmoothing(self, image, size, sigma):
        kernel = cv.getGaussianKernel(size, sigma)
        kernel = np.outer(kernel, kernel)
        S = sci.convolve2d(kernel, image)

        return S

    def ImageGradient(self, S, method):
        if method == 'sobel':
            Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        elif method == 'robert':
            Gx = np.array([[1, 0], [0, -1]])
            Gy = np.array([[0, -1], [1, 0]])
        elif method == 'prewitt':
            Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Gy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        
        Ix = sci.convolve2d(Gx, S)
        Iy = sci.convolve2d(Gy, S)

        theta = np.arctan2(Iy, Ix)
        mag = np.sqrt(Iy**2 + Ix**2)
        mag = mag/mag.max()*255

        return mag.astype(int), theta

    def FindThreshold(self, Mag, percentageOfNonEdge):
        hist, bins = np.histogram(Mag.flatten(), 256)
        pdf = hist/sum(hist)
        cdf = np.cumsum(pdf)
        high_index = np.argmax(cdf > percentageOfNonEdge)

        T_high = high_index
        T_low = 0.5*T_high

        return T_low, T_high

    def NonmaximaSupress(self, Mag, Theta):
        row, col = Mag.shape
        New_Mag = Mag*0
        angle = Theta * 180 / np.pi

        angle[angle < 0] =+ 180

        for i in range(1,row-1):
            for j in range(1,col-1):
                x = 255
                y = 255
                
                #angle 0
                if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                    x = Mag[i, j+1]
                    y = Mag[i, j-1]
                #angle 45
                elif (22.5 <= angle[i][j] < 67.5):
                    x = Mag[i+1, j-1]
                    y = Mag[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i][j] < 112.5):
                    x = Mag[i+1, j]
                    y = Mag[i-1, j]
                #angle 135
                elif (112.5 <= angle[i][j] < 157.5):
                    x = Mag[i-1, j-1]
                    y = Mag[i+1, j+1]

                if (Mag[i,j] >= x) and (Mag[i,j] >= y):
                    New_Mag[i,j] = Mag[i,j]
                else:
                    New_Mag[i,j] = 0
        
        return New_Mag

    def EdgeLinking(self, New_Mag, T_low, T_high):
        
        weakval = 10
        strongval = 255

        strong = New_Mag*0
        weak = New_Mag*0

        row, col = np.shape(New_Mag)

        for i in range(row):
            for j in range(col):
                if New_Mag[i][j] >= T_high:
                    strong[i][j] = strongval
                if New_Mag[i][j] >= T_low:
                    weak[i][j] = weakval

        plt.subplot(2,4,5)
        plt.title('Weak')
        plt.imshow(weak, cmap='gray')

        plt.subplot(2,4,6)
        plt.title('Strong')
        plt.imshow(strong, cmap='gray')
 
        New_Mag_copy = New_Mag.copy()

        for i in range(1, row-1):
            for j in range(1, col-1):
                if weak[i,j] == weakval:
                    if ((strong[i+1, j-1] == strongval) or (strong[i+1, j] == strongval) or (strong[i+1, j+1] == strongval)
                        or (strong[i, j-1] == strongval) or (strong[i, j+1] == strongval)
                        or (strong[i-1, j-1] == strongval) or (strong[i-1, j] == strongval) or (strong[i-1, j+1] == strongval)):
                        New_Mag[i, j] = strongval
                    elif ((New_Mag_copy[i+1, j-1] == strongval) or (New_Mag_copy[i+1, j] == strongval) or (New_Mag_copy[i+1, j+1] == strongval)
                        or (New_Mag_copy[i, j-1] == strongval) or (New_Mag_copy[i, j+1] == strongval)
                        or (New_Mag_copy[i-1, j-1] == strongval) or (New_Mag_copy[i-1, j] == strongval) or (New_Mag_copy[i-1, j+1] == strongval)):
                        New_Mag[i, j] = strongval
                    else:
                        New_Mag_copy[i, j] = 0

        return New_Mag_copy
    
    def check_neighbors(self, i, j, array):
        if (array[i+1][j] != 0) or (array[i-1][j] != 0) or (array[i][j+1] != 0) or (array[i][j-1] != 0) or (array[i+1][j+1] != 0) or (array[i-1][j-1] != 0) or (array[i-1][j+1] != 0) or (array[i+1][j-1] != 0):
            return True
        else:
            return False

# MAIN CODE ------------------------------------------------------------------

# Initialize class
Canny = Canny()

# Load image (options are lena.bmp, test1.bmp, pointer1.bmp, or joy1.bmp)
image = cv.imread('/home/codynichoson/Q1/computer_vision/5 - Canny Edge Detection/lena.bmp', cv.IMREAD_GRAYSCALE)

# Gaussian smoothing
S = Canny.GaussSmoothing(image, 5, 1)

# Find image gradient magnitude and angle
Mag, Theta = Canny.ImageGradient(S, 'sobel') # 'sobel', 'robert', or 'prewitt'

# Find high and low thresholds
T_low, T_high = Canny.FindThreshold(Mag, 0.8)

# Apply non-maximum suppression
New_Mag = Canny.NonmaximaSupress(Mag, Theta)

# Link edges
result = Canny.EdgeLinking(New_Mag, T_low, T_high)

# Plot results
plt.figure(1)
plt.subplot(2,4,1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(2,4,2)
plt.title('Gaussian Smoothed')
plt.imshow(S, cmap='gray')
plt.subplot(2,4,3)
plt.title('Gradient Mag')
plt.imshow(Mag, cmap='gray')
plt.subplot(2,4,4)
plt.title('Nonmaxima Suppressed')
plt.imshow(result, cmap='gray')
plt.subplot(2,4,7)
plt.title('Edge Linked')
plt.imshow(New_Mag, cmap='gray')

plt.show()

# This code will execute several morphological operators
# Specifically, Erosion and Dilation, plus using them to do
# Opening, Closing, and Boundaries

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2 as cv
from IPython.display import display
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

# Import images
#img1 = cv.imread('/home/codynichoson/computer_vision/MP2/gun.bmp',0)
img1 = cv.imread('/home/codynichoson/computer_vision/MP2/palm.bmp',0)
height = img1.shape[0]
width = img1.shape[1]
img1_bin = img1/255

# Plot original image
plt.subplot(3, 3, 1)
plt.figure(1)
plt.imshow(img1_bin)
plt.title("Original")

def dilation(img, size):
    se = [size,size]

    x = se[0] // 2
    y = se[1] // 2

    height = img.shape[0]
    width = img.shape[1]

    new_img = np.zeros((height,width))

    for r in range(height):
        for c in range(width):
            for i in range(-x,x+1):
                for j in range(-y,y+1):
                    if img[r][c]:
                        new_img[r+i][c+j] = 1
                    else:
                        pass

    return new_img
    
def erosion(img, size):
    se = [size,size]
    SE = np.array([[1,1,1],[1,1,1],[1,1,1]])

    x = se[0] // 2
    y = se[1] // 2

    height = img.shape[0]
    width = img.shape[1]

    new_img = np.zeros((height,width))

    for r in range(height-1):
        for c in range(width-1):
            elem = get_element(img, r, c)
            if elem.all()== SE.all():
                new_img[r][c] = 1

    return new_img

def opening(img, SE):
    x = erosion(img, SE)
    open_img = dilation(x, SE)

    return open_img

def closing(img, SE):
    x = dilation(img, SE)
    close_img = erosion(x, SE)

    return close_img

def boundary(img):
    x = closing(img,5)
    bound_img = x - erosion(x,5)
    #bound_img = dilation(img, 3) - img1_bin

    return bound_img

def get_element(img, r, c):
    element = np.array([[0,0,0],[0,0,0],[0,0,0]])
    element[1][1] = img[r][c]
    element[0][1] = img[r-1][c]
    element[2][1] = img[r+1][c]
    element[1][0] = img[r][c-1]
    element[1][2] = img[r][c+1]
    element[2][2] = img[r+1][c+1]
    element[0][0] = img[r-1][c-1]
    element[2][0] = img[r+1][c-1]
    element[0][2] = img[r-1][c+1]

    return element

# Run the functions
dil3_img = dilation(img1,3)
dil5_img = dilation(img1,5)
ero3_img = erosion(img1,3)
ero5_img = erosion(img1,5)
open3_img = opening(img1,3)
open5_img = opening(img1,5)
close3_img = closing(img1,3)
bound_img = boundary(img1)

# Plot the images
plt.subplot(3, 3, 2)
plt.imshow(dil3_img)
plt.title("Dilation 3x3")

plt.subplot(3, 3, 3)
plt.imshow(dil5_img)
plt.title("Dilation 5x5")

plt.subplot(3, 3, 4)
plt.imshow(ero3_img)
plt.title("Erosion 3x3")

plt.subplot(3, 3, 5)
plt.imshow(ero5_img)
plt.title("Erosion 5x5")

plt.subplot(3, 3, 6)
plt.imshow(open3_img)
plt.title("Opening 3x3")

plt.subplot(3, 3, 7)
plt.imshow(open5_img)
plt.title("Opening 5x5")

plt.subplot(3, 3, 8)
plt.imshow(open3_img)
plt.title("Closing 3x3")

plt.subplot(3, 3, 9)
plt.imshow(bound_img)
plt.title("Boundary")

plt.show()

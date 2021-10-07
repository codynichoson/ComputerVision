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
img1 = cv.imread('/home/codynichoson/computer_vision/MP2/gun.bmp',0)

plt.figure(1)
img_show = plt.imshow(img1)

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

    plt.figure(2)
    img_show = plt.imshow(new_img)
    

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

    plt.figure(3)
    img_show = plt.imshow(new_img)

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

def opening(img, SE):
    pass

def closing(img, SE):
    pass

def boundary(img):
    pass


t = erosion(img1,3)

plt.show()

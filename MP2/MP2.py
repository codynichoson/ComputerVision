# This code will execute several morphological operators
# Specifically, Erosion and Dilation, plus using them to do
# Opening, Closing, and Boundaries

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2 as cv
from IPython.display import display
np.set_printoptions(threshold=sys.maxsize)

# Import images
img1 = Image.open('/home/codynichoson/computer_vision/MP2/gun.bmp')


# Initialize variables
width, height = img1.size

SE = [[255,255,255], [255,255,255], [255,255,255]]

def dilation(img, SE):
    pixelMap = img.load()

    img_new = Image.new(img.mode, img.size)
    pixelsNew = img_new.load()

    for x in range(SE):
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                if pixelMap[i,j] == 255:
                    pixelsNew[i-1,j] = 255
                    pixelsNew[i+1,j] = 255
                    pixelsNew[i,j-1] = 255
                    pixelsNew[i,j+1] = 255
                    pixelsNew[i+1,j+1] = 255
                    pixelsNew[i-1,j-1] = 255
                    pixelsNew[i+1,j-1] = 255
                    pixelsNew[i-1,j+1] = 255
                else:
                    pixelsNew[i,j] = pixelMap[i,j]

    img_new.show()

def erosion(img, SE):
    pass

def opening(img, SE):
    pass

def closing(img, SE):
    pass

def boundary(img):
    pass

img1.show()
dilation(img1, 1)
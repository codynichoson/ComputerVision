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
img1 = Image.open('/home/codynichoson/computer_vision/MP2/gun.bmp')

img2 = cv.imread('/home/codynichoson/computer_vision/MP2/gun.bmp',0)

# Initialize variables
width, height = img1.size


def dilation(img, size):
    se = [size,size]
    print(se)

    x = se[0] // 2
    y = se[1] // 2
    print(x)

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

    plt.figure(1)
    img_show = plt.imshow(new_img)
    

def erosion(img, size):
    pass

def opening(img, SE):
    pass

def closing(img, SE):
    pass

def boundary(img):
    pass

#img1.show()
#dilation(img1)
#dilation2(img2)

t = dilation(img2,3)

plt.show()

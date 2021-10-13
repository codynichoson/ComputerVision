from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2 as cv
from IPython.display import display
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

# Function for using HSV colorspace to detect skin
def hsv_skin_detection(img, sample, row):
    height = img.shape[0]
    width = img.shape[1]
    sample = cv.cvtColor(sample, cv.COLOR_BGR2HSV)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    h= sample[:,:,0]
    hist_h = cv.calcHist([h],[0],None,[256],[0,256])

    # Find thresholds
    max_h = np.max(h)
    min_h = np.min(h)

    for r in range(height):
        for c in range(width):
            if hsv_img[r,c,0] > max_h or img[r,c,0] < min_h:
                hsv_img[r,c,0] = 0
                hsv_img[r,c,1] = 0
                hsv_img[r,c,2] = 0
                
    # Plot figures
    plt.figure('HSV Skin-Tone Detection')
    plt.subplot(3, 4, 1 - (1-row)*4)
    plt.imshow(rgb_img)
    plt.title("Original Image")

    plt.subplot(3, 4, 2 - (1-row)*4)
    plt.imshow(cv.cvtColor(sample, cv.COLOR_HSV2RGB))
    plt.title("Cropped Image")

    plt.subplot(3, 4, 3 - (1-row)*4)
    plt.plot(hist_h)
    plt.title("Histogram (Hue)")

    plt.subplot(3, 4, 4 - (1-row)*4)
    plt.imshow(cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB))
    plt.title("Resulting Image")

# Function for using RGB colorspace to detect skin
def rgb_skin_detection(img, sample, row):
    height = img.shape[0]
    width = img.shape[1]

    sample = cv.cvtColor(sample, cv.COLOR_BGR2RGB)

    original = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    r, g, b = sample[:,:,0], sample[:,:,1], sample[:,:,2]
    hist_r = cv.calcHist([r],[0],None,[256],[0,256])
    hist_g = cv.calcHist([g],[0],None,[256],[0,256])
    hist_b = cv.calcHist([b],[0],None,[256],[0,256])

    # Find thresholds
    max_r = np.max(r) + 20
    min_r = np.min(r) - 20
    max_g = np.max(g)
    min_g = np.min(g)
    max_b = np.max(b) + 10
    min_b = np.min(b) - 10

    for r in range(height):
        for c in range(width):
            if rgb_img[r,c,0] > max_r or img[r,c,0] < min_r or rgb_img[r,c,1] > max_g or img[r,c,1] < min_g or rgb_img[r,c,2] > max_b or img[r,c,2] < min_b:
                rgb_img[r,c,0] = 0
                rgb_img[r,c,1] = 0
                rgb_img[r,c,2] = 0
                
    # Plot figures
    plt.figure('RGB Skin-Tone Detection')
    plt.subplot(3, 4, 1 - (1-row)*4)
    plt.imshow(original)
    plt.title("Original Image")

    plt.subplot(3, 4, 2 - (1-row)*4)
    plt.imshow(sample)
    plt.title("Cropped Image")

    plt.subplot(3, 4, 3 - (1-row)*4)
    plt.plot(hist_r, color='r')
    plt.plot(hist_g, color='g')
    plt.plot(hist_b, color='b')
    plt.title("Histogram (RGB)")

    plt.subplot(3, 4, 4 - (1-row)*4)
    plt.imshow(rgb_img)
    plt.title("Resulting Image")

# Import sample images and select their skin tone sample crops using interactive GUI
gun_img = cv.imread('/home/codynichoson/Q1/computer_vision/4 - Histogram-based Skin Color Detection/gun1.bmp')
bounds1 = cv.selectROI(gun_img)
gun_sample = gun_img[bounds1[1]:bounds1[1]+bounds1[3], bounds1[0]:bounds1[0]+bounds1[2]]

joy_img = cv.imread('/home/codynichoson/Q1/computer_vision/4 - Histogram-based Skin Color Detection/joy1.bmp')
bounds2 = cv.selectROI(joy_img)
joy_sample = joy_img[bounds2[1]:bounds2[1]+bounds2[3], bounds2[0]:bounds2[0]+bounds2[2]]

pointer_img = cv.imread('/home/codynichoson/Q1/computer_vision/4 - Histogram-based Skin Color Detection/pointer1.bmp')
bounds3 = cv.selectROI(pointer_img)
pointer_sample = pointer_img[bounds3[1]:bounds3[1]+bounds3[3], bounds3[0]:bounds3[0]+bounds3[2]]

# Run HSV histogram-based skin color detection
hsv_skin_detection(gun_img, gun_sample, 1)
hsv_skin_detection(joy_img, joy_sample, 2)
hsv_skin_detection(pointer_img, pointer_sample, 3)

# Run RGB histogram-based skin color detection
rgb_skin_detection(gun_img, gun_sample, 1)
rgb_skin_detection(joy_img, joy_sample, 2)
rgb_skin_detection(pointer_img, pointer_sample, 3)

plt.show()
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2 as cv
from IPython.display import display
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

def histo(img, bins):
    histogram = np.zeros(bins)
    for pixel in img:
        histogram[pixel] = histogram[pixel] + 1
    return histogram

# Import and flatten image
img = cv.imread('/home/codynichoson/Q1/computer_vision/3 - Histogram Equalization/moon.bmp',1)
flat_img = img.flatten()

fig = plt.figure()

# Create histogram
histogram = histo(flat_img, 256)
plt.subplot(2, 3, 1)
plt.plot(histogram)
plt.title("Original Histo")

# Calculating and normalizing to proper range
cdf = histogram.cumsum()
num = (cdf - cdf.min())*255
den = cdf.max() - cdf.min()
cdf = num/den
cdf = cdf.astype('uint8') # Need int for images
plt.subplot(2, 3, 2)
plt.plot(cdf)
plt.title("Original CDF")

# Use flat_img values as index to find corresponding value in cumsum
new_img = cdf[flat_img]
new_img = np.reshape(new_img, img.shape) #reshape ot proper size

# Make histogram and cdf of edited image
histogram2 = histo(new_img, 256)
plt.subplot(2, 3, 4)
plt.plot(histogram2)
plt.title("New Histo")
cdf2 = histogram2.cumsum()
plt.subplot(2, 3, 5)
plt.plot(cdf2)
plt.title("New CDF")

# Plot images
plt.subplot(2, 3, 3)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(2, 3, 6)
plt.imshow(new_img)
plt.title("New Image")

plt.show()

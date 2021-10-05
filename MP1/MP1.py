# This code will run a simple Connected Component Labeling (CCL) method on several test images

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# Import images
img1 = Image.open('/home/codynichoson/computer_vision/MP1/test.bmp')
img2 = Image.open('/home/codynichoson/computer_vision/MP1/face.bmp')
img3 = Image.open('/home/codynichoson/computer_vision/MP1/gun.bmp')
img4 = Image.open('/home/codynichoson/computer_vision/MP1/face_old.bmp')

def run_ccl(img):
    # Initialize variables
    E_table = []
    label = 0
    width, height = img.size
    binary_arr = np.zeros([height, width], dtype=int)

    # First pass to make initial labels
    for row in range(height):
        for column in range(width):
            if img.getpixel((column,row)) == 255:
                L_upper = binary_arr[row][column-1] # Upper label
                L_left = binary_arr[row-1][column] # Left label
                if L_upper == L_left and L_upper != 0 and L_left != 0:
                    binary_arr[row][column] = L_upper
                elif L_upper != L_left and (L_upper == 0 or L_left == 0):
                    binary_arr[row][column] = max(L_upper, L_left)
                elif L_upper != L_left and L_upper > 0 and L_left > 0:
                    binary_arr[row][column] = min(L_upper, L_left)
                    E_table.append(tuple([L_upper, L_left]))
                else:
                    label = label + 1
                    binary_arr[row][column] = label 
    
    # Process E_table
    E_table_set = set(E_table) # Get rid of duplicates
    E_table = list(E_table_set) # Convert back to list
    for x in E_table: 
        E_table[E_table.index(x)] = sorted(x, reverse=True) # Put pairs in order (larger then smaller)
    E_table.sort(reverse=True) # Sort by first element (reverse order)

    # Second pass to apply equal pairs from E_table
    for row in range(height):
        for column in range(width):
            if binary_arr[row][column] != 0: # If the pixel has value other than 0
                for pair in E_table: # Check E_table
                    if binary_arr[row][column] == pair[0]: # If the current pixel equals the first number in E pair
                        binary_arr[row][column] = pair[1] # Then change pixel value to second number in E pair

    return binary_arr

# Run CCL function for all four images
binary_arr1 = run_ccl(img1)
binary_arr2 = run_ccl(img2) 
binary_arr3 = run_ccl(img3) 
binary_arr4 = run_ccl(img4) 

# Plot resulting array for each image
plot1 = plt.figure(1)
img_show = plt.imshow(binary_arr1)
plot2 = plt.figure(2)
img_show = plt.imshow(binary_arr2)
plot3 = plt.figure(3)
img_show = plt.imshow(binary_arr3)
plot4 = plt.figure(4)
img_show = plt.imshow(binary_arr4)

plt.show()

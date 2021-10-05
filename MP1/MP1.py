# This code will run a simple Connected Component Labeling (CCL) method on several test images

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

img = Image.open('/home/codynichoson/Computer_Vision/MP1/face.bmp')


width = 253
height = 231

binary_arr = np.zeros([height, width], dtype=int)

label = 0

E_table = []

# First pass to make initial labels
for row in range(height):
    for column in range(width):
        if img.getpixel((column,row)) == 255:
            L_upper = binary_arr[row][column-1] # Upper label
            L_left = binary_arr[row-1][column] # Left label
            if L_upper == L_left and L_upper != 0 and L_left != 0: # Same label
                binary_arr[row][column] = L_upper
            elif L_upper != L_left and (L_upper == 0 or L_left == 0): # Left neighbor
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
E_table.sort() # Sort by first element


for row in range(height):
    for column in range(width):
        if binary_arr[row][column] != 0:
            L_upper = binary_arr[row][column-1] # Upper label
            L_left = binary_arr[row-1][column] # Left label
        #if current pixel has value that != 0
        #for (iterate e table)



#print(E_table)

plot1 = plt.figure(1)
img_show = plt.imshow(binary_arr)
plt.show()

# Second scanning
# Renumber labels using E_table







# Check pixel, if value = 255, we label pixel

# If pixel labeled, check neighbors above and to left (A & B)
    # If pixels above or left are also 255, apply same label
    # If both are 0, then create new label and store 

# If A & B are different, assign smaller value to current pixel
    # Also change larger value of A & B to be the smaller value

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

global directory, ssd_directory, cc_directory, ncc_directory

directory = '/home/codynichoson/Q1/computer_vision/7 - Head Tracking'
test_directory = '/home/codynichoson/Q1/computer_vision/7 - Head Tracking/image_girl'
ssd_directory = '/home/codynichoson/Q1/computer_vision/7 - Head Tracking/ssd_results'
cc_directory = '/home/codynichoson/Q1/computer_vision/7 - Head Tracking/cc_results'
ncc_directory = '/home/codynichoson/Q1/computer_vision/7 - Head Tracking/ncc_results'

# Take method as an input to determine which is used
method = input('Enter matching method (ssd, cc, or ncc): ')

def TempMatchFolder(template):
    """ Function to iterate through sample video frames """
    for filename in sorted(os.listdir(test_directory)):
        if filename.endswith(".jpg"):
            # Read image
            img = cv2.imread('%s/%s' %(test_directory,filename))

            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Run template matching
            result = TemplateMatching(img_gray, template)

            # Draw rectangle on frame at result
            cv2.rectangle(img, (result[0], result[1]), (result[0] + w, result[1] + h), (200, 0, 200), 2)

            # Save frame with rectangle
            if method == 'ssd':
                cv2.imwrite('%s/%s' %(ssd_directory,filename), img)
            elif method == 'cc':
                cv2.imwrite('%s/%s' %(cc_directory,filename), img)
            elif method == 'ncc':
                cv2.imwrite('%s/%s' %(ncc_directory,filename), img)

def TemplateMatching(img, template):
    """ Function to run template matching methods """
    # Initializations
    img_h, img_w = img.shape[0], img.shape[1]
    temp_h, temp_w = template.shape[0], template.shape[1]
    score = np.empty((img_h - temp_h, img_w - temp_w))

    # Iterate through image and determine template matching score
    for row in range(0, img_h - temp_h):
        for col in range(0, img_w - temp_w):
            if method == 'ssd':
                difference = (np.abs(img[row:row + temp_h, col:col + temp_w] - template))**2
            elif method == 'cc':
                difference = (np.abs(img[row:row + temp_h, col:col + temp_w]*(template)))
            elif method == 'ncc':
                img_avg = np.mean(img)
                temp_avg = np.mean(template)
                I_hat = (img[row:row + temp_h, col:col + temp_w]) - img_avg
                T_hat = template-temp_avg
                num = I_hat*T_hat
                den = np.sqrt((I_hat**2)*(T_hat**2))
                difference = -num/den
            
            score[row, col] = difference.sum()

    # Determine result
    result = np.unravel_index(score.argmin(), score.shape)

    return(result[1], result[0])

# Use GUI to choose template
img = cv2.imread('/home/codynichoson/Q1/computer_vision/7 - Head Tracking/image_girl/0001.jpg')
cv2.namedWindow("Selection", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Selection", 400, 400)  
bounds = cv2.selectROI('Selection', img)
template = img[bounds[1]:bounds[1]+bounds[3], bounds[0]:bounds[0]+bounds[2]]

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

h, w = template.shape

# Run TempMatchFolder function
TempMatchFolder(template)

# Create videos from solution frames
img_array = []
if method == 'ssd':
    for filename in sorted(glob.glob('%s/*.jpg' %(ssd_directory))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
elif method == 'cc':
    for filename in sorted(glob.glob('%s/*.jpg' %(cc_directory))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
elif method == 'ncc':
    for filename in sorted(glob.glob('%s/*.jpg' %(ncc_directory))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
if method == 'ssd':
    out = cv2.VideoWriter('%s/ssd_results.mp4' %(directory),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
elif method == 'cc':
    out = cv2.VideoWriter('%s/cc_results.mp4' %(directory),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
elif method == 'ncc':
    out = cv2.VideoWriter('%s/ncc_results.mp4' %(directory),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def hough_transform(image, edge_image, rho_max = 180, theta_max = 180, threshold = 100):
  # Get dimensions
  edge_row, edge_col = edge_image.shape
  
  # Determine theta values
  dtheta = 180 / theta_max
  theta_vals = np.arange(0, 180, dtheta)
  cos = np.cos(np.deg2rad(theta_vals))
  sin = np.sin(np.deg2rad(theta_vals))

  # Determine rho values
  rho_range = np.sqrt(edge_row**2 + edge_col**2)
  drho = (2*rho_range) / rho_max
  rho_vals = np.arange(-rho_range, rho_range, drho)

  # Initialize accumulator
  accumulator = np.zeros((len(rho_vals), len(rho_vals)))

  # Setting up plotting
  fig = plt.figure(1)
  plot1 = fig.add_subplot(2,3,1)
  plot2 = fig.add_subplot(2,3,2)
  plot3 = fig.add_subplot(2,3,3)
  plot4 = fig.add_subplot(2,3,4)

  plot1.imshow(image)
  plot2.imshow(edge_image, cmap='gray')
  plot3.set_facecolor((0,0,0))
  plot4.imshow(image)

  plot1.title.set_text('Original Image')
  plot2.title.set_text('Canny Edges')
  plot3.title.set_text('Parameter Space')
  plot4.title.set_text('Detected Lines')
  
  # Parameter space loops
  for i in range(edge_row):
    for j in range(edge_col):
      if edge_image[i][j] != 0:
        edge_point = [i - edge_row/2, j - edge_col/2]
        x_param, y_param = [], []
        for th in range(len(theta_vals)):
          rho = (edge_point[1] * cos[th]) + (edge_point[0] * sin[th])
          theta = theta_vals[th]
          rh = np.argmin(np.abs(rho_vals - rho))
          accumulator[rh][th] = accumulator[rh][th] + 1
          y_param.append(rho)
          x_param.append(theta)
        plot3.plot(x_param, y_param, color = 'white', alpha = 0.03)

  # Line determining loops
  for i in range(accumulator.shape[0]):
    for j in range(accumulator.shape[1]):
      if accumulator[i][j] > threshold:
        rho = rho_vals[i]
        theta = theta_vals[j]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a*rho) + edge_col/2
        y0 = (b*rho) + edge_row/2
        x1 = int(x0 + 1000*-b)
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*-b)
        y2 = int(y0 - 1000*a)
        plot4.add_line(mlines.Line2D([x1, x2], [y1, y2], color='purple'))

if __name__ == "__main__":
    # Import image
    image = cv.imread('/home/codynichoson/Q1/computer_vision/6 - Hough Transform/input.bmp')

    # Use Canny to detect edges
    edge_image = cv.Canny(image, 60, 240, 5)

    # Use Hough transform to find lines
    hough_transform(image, edge_image)
    
    plt.show()
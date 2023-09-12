import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/sambaran/Pictures/jackal_sim.png')  # Replace 'your_image.jpg' with the path to your image

# Define the lower and upper threshold for the black color (you may need to adjust these)
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([10, 10, 10], dtype=np.uint8)

# Create a mask to identify the black background
mask = cv2.inRange(image, lower_black, upper_black)

# Replace the black background with white
image[mask > 0] = [255, 255, 255]  # White color in BGR format
image = cv2.resize(image, (1024, 900), cv2.INTER_AREA)

# Save the resulting image
cv2.imwrite('/home/sambaran/Pictures/jackal_white_sim.png', image)  # Replace 'output_image.jpg' with your desired output file name
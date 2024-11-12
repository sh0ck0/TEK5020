import cv2
import matplotlib.pyplot as plt

# Load the training image
image_train_path = 'Bilde2.png'
image_train = cv2.imread(image_train_path)
image_train_rgb = cv2.cvtColor(image_train, cv2.COLOR_BGR2RGB)

# Define the regions used for training
# Adjust these coordinates if needed based on your training regions
blue_region_coords = (2000, 400, 100, 100)  # (x, y, width, height) for blue folder
red_region_coords = (700, 900, 100, 100)  # (x, y, width, height) for red folder
background_region_coords = (10, 10, 50, 50)  # (x, y, width, height) for background

# Draw rectangles on the image to show the training regions
image_with_regions = image_train_rgb.copy()
cv2.rectangle(image_with_regions, (blue_region_coords[0], blue_region_coords[1]), 
              (blue_region_coords[0] + blue_region_coords[2], blue_region_coords[1] + blue_region_coords[3]), 
              (0, 255, 0), 3)  # Green rectangle for blue folder region

cv2.rectangle(image_with_regions, (red_region_coords[0], red_region_coords[1]), 
              (red_region_coords[0] + red_region_coords[2], red_region_coords[1] + red_region_coords[3]), 
              (255, 0, 0), 3)  # Red rectangle for red folder region

cv2.rectangle(image_with_regions, (background_region_coords[0], background_region_coords[1]), 
              (background_region_coords[0] + background_region_coords[2], background_region_coords[1] + background_region_coords[3]), 
              (0, 0, 255), 3)  # Blue rectangle for background region

# Display the image with marked training regions
plt.figure(figsize=(8, 8))
plt.title("Training Regions Highlighted")
plt.imshow(image_with_regions)
plt.axis('off')
plt.show()


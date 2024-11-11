import cv2
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load images
train_image_path = 'Bilde2.png'  # Replace with actual path
test_image_path = 'Bilde3.png'   # Replace with actual path
train_image = cv2.imread(train_image_path)
test_image = cv2.imread(test_image_path)

# Function to extract RGB data from selected regions
def get_rgb_data(image, regions):
    data = []
    labels = []
    for label, (x, y, w, h) in regions.items():
        for i in range(x, x + w):
            for j in range(y, y + h):
                rgb = image[j, i, :3]  # Extract RGB values
                data.append(rgb)
                labels.append(label)
    return np.array(data), np.array(labels)

# Define regions for training data as (x, y, width, height)
# Adjust these regions as needed for better segmentation
regions = {
    0: (2000, 400, 900, 900),  # Region for red folder
    1: (400, 900, 900, 900),  # Region for blue folder
    2: (10, 10, 750, 750)     # Region for background
}

# Get training data from the training image
train_data, train_labels = get_rgb_data(train_image, regions)

# Normalize RGB values to account for lighting variations
train_data_normalized = train_data / (train_data.sum(axis=1, keepdims=True) + 1e-6)

# Train an SVM classifier
svm = SVC(kernel='linear')  # Try 'rbf' kernel if needed
svm.fit(train_data_normalized, train_labels)

# Function to classify each pixel in the test image
def segment_image(image, classifier):
    # Normalize RGB values of the test image
    image_normalized = image / (image.sum(axis=2, keepdims=True) + 1e-6)
    segmented_image = np.zeros(image.shape[:2], dtype=int)
    
    # Flatten the image array for classification
    reshaped_data = image_normalized.reshape(-1, 3)
    predictions = classifier.predict(reshaped_data)
    
    # Reshape the predictions back to the original image shape
    segmented_image = predictions.reshape(image.shape[:2])
    return segmented_image

# Segment the test image
segmented_test_image = segment_image(test_image, svm)

# Display the original and segmented images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_test_image, cmap='viridis')

plt.show()


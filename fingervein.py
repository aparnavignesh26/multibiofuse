from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import cv2
import os

# Define the path to the directory containing the mmcbnu_6000 dataset
dataset_path = '/content/drive/MyDrive/MMCBNU_6000'

# Function to preprocess a single image using histogram equalization
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    return equalized_image

# Load the mmcbnu_6000 dataset
image_list = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            image_list.append(image)

# Preprocess each image in the dataset using histogram equalization
preprocessed_images = []
for image in image_list:
    preprocessed_image = preprocess_image(image)
    preprocessed_images.append(preprocessed_image)

# Convert the list of preprocessed images to a NumPy array
preprocessed_dataset = np.array(preprocessed_images)

# Save the preprocessed dataset
np.save('preprocessed_finger_vein_dataset.npy', preprocessed_dataset)



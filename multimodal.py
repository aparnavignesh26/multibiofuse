#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from sklearn.metrics import pairwise_distances
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

fingerprint_model = load_model('fingerprint_model.h5')  # Replace with your fingerprint model
finger_vein_model = load_model('vein_model.h5')  # Replace with your finger vein model
face_model = load_model('face_model.h5')  # Replace with your face recognition model

face_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/facenew/Aaron_Patterson/Aaron_Patterson_0001.jpg' 
fingerprint_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/fp/1/1__M_Left_index_finger.BMP'
finger_vein_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/desiredfv/001/L_Fore/01.bmp'


def preprocess_fingerprint(fingerprint_image_path):
    image = cv2.imread(fingerprint_image_path)  # Load the image from the provided path
    
    if image is None or image.size == 0:
        # Handle the case where the image loading fails or it's empty
        print(f"Warning: Failed to load or empty fingerprint image at path: {fingerprint_image_path}")
        return None

    # Resize the fingerprint image to the desired size (e.g., 100x100)
    image = cv2.resize(image, (100, 100))
   
    # Normalize pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def preprocess_finger_vein(finger_vein_image_path, target_size=(100, 100), is_grayscale=True):
    # Load the image
    finger_vein_image = cv2.imread(finger_vein_image_path)
    
    # Check if the image was loaded successfully
    if finger_vein_image is not None:
        # Resize the image to the target size
        resized_image = cv2.resize(finger_vein_image, target_size)
        
        if is_grayscale:
            # Convert the resized image to grayscale
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            # Expand dimensions to add the single channel
            final_image = np.expand_dims(gray_image, axis=-1)
        else:
            # Normalize pixel values to the range [0, 1] for color images
            final_image = resized_image.astype(np.float32) / 255.0
            
        return final_image
    else:
        print(f"Failed to load the image: {finger_vein_image_path}")
        return None
    
def preprocess_facial(face_image_path, target_size=(100, 100), is_grayscale=False):
    # Load the image
    image = cv2.imread(face_image_path)
    
    # Check if the image was loaded successfully
    if image is not None:
        # Resize the image to the target size
        resized_image = cv2.resize(image, target_size)
        
        if is_grayscale:
            # Convert the resized image to grayscale
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            # Expand dimensions to add the single channel
            final_image = np.expand_dims(gray_image, axis=-1)
        else:
            # Normalize pixel values to the range [0, 1] for color images
            final_image = resized_image.astype(np.float32) / 255.0
            
        return final_image

# Load and preprocess datasets for each modality
face_images = [preprocess_facial(face_image_path)]
vein_images = [preprocess_finger_vein(finger_vein_image_path)]
fingerprint_images = [preprocess_fingerprint(fingerprint_image_path)]


def extract_features(image, model):
    # Use the provided model to extract features from the image
    features = model.predict(np.expand_dims(image, axis=0))
    return features

# Extract features for each modality (you may use different feature extraction methods)
face_features = [extract_features(face_images, face_model)]
vein_features = [extract_features(vein_images, vein_model)]
fingerprint_features = [extract_features(fingerprint_images, fingerprint_model)]

# Calculate matching scores for each modality
face_matching_scores = pairwise_distances(face_features, test_face_features, metric='cosine')
vein_matching_scores = pairwise_distances(vein_features, test_vein_features, metric='cosine')
fingerprint_matching_scores = pairwise_distances(fingerprint_features, test_fingerprint_features, metric='cosine')

# Fusion: Combine matching scores from each modality (e.g., simple average fusion)
combined_scores = (face_matching_scores + vein_matching_scores + fingerprint_matching_scores) / 3

# Set a threshold for matching decision
threshold = 0.6

# Make identity decision based on the combined matching scores
matches = combined_scores < threshold


# In[10]:


import numpy as np
from sklearn.metrics import accuracy_score

# Generate example similarity scores for each modality (replace with your actual scores)
face_scores = np.random.rand(100)
vein_scores = np.random.rand(100)
fingerprint_scores = np.random.rand(100)

# Define a threshold for decision-making (you can fine-tune this)
threshold = 0.5

# Create ground truth labels for genuine and impostor pairs
genuine_labels = np.ones(100)  # All pairs are genuine
impostor_labels = np.zeros(100)  # All pairs are impostor

# Concatenate scores from different modalities (you can use fusion methods here)
# For example, you can take the maximum, minimum, average, or any other fusion method
# Here, we'll use the average of scores as an example fusion method
fusion_scores = (face_scores + vein_scores + fingerprint_scores) / 3

# Combine ground truth labels for genuine and impostor pairs
labels = np.concatenate([genuine_labels, impostor_labels])

# Make binary decisions based on the threshold
decisions = fusion_scores >= threshold

# Calculate accuracy
accuracy = accuracy_score(labels, decisions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate FAR and FRR
# FAR: False Acceptance Rate (Impostors incorrectly accepted)
# FRR: False Rejection Rate (Genuine users incorrectly rejected)
impostor_indices = np.where(genuine_labels == 0)[0]
genuine_indices = np.where(genuine_labels == 1)[0]

FAR = np.sum(decisions[impostor_indices] == 1) / len(impostor_indices)
FRR = np.sum(decisions[genuine_indices] == 0) / len(genuine_indices)

print(f"FAR: {FAR * 100:.2f}%")
print(f"FRR: {FRR * 100:.2f}%")

# Fine-tune system parameters and fusion methods as needed
# You can vary the threshold, try different fusion methods, or optimize other parameters


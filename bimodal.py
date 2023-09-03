#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
fingerprint_model = load_model('fingerprint_model.h5')  # Replace with your fingerprint model
finger_vein_model = load_model('vein_model.h5')  # Replace with your finger vein model
def preprocess_fingerprint(image):
    if image is None or image.size == 0:
        # Handle the case where the image loading fails or it's empty
        print("Warning: Failed to load or empty fingerprint image.")
        return None

    # Resize the fingerprint image to the desired size (e.g., 224x224)
    image = cv2.resize(image, (100, 100))
   
    # Normalize pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0

    return image

def preprocess_finger_vein(image):
    if image is None or image.size == 0:
        print("Warning: Failed to load or empty finger vein image.")
        return None

    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the finger vein image to the desired size (e.g., 100x100)
    gray_image = cv2.resize(gray_image, (100, 100))

    # Normalize pixel values to the range [0, 1]
    gray_image = gray_image.astype(np.float32) / 255.0

    # Expand dimensions to match the model's input shape
    gray_image = np.expand_dims(gray_image, axis=-1)

    return gray_image

from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(features1, features2):
    # Calculate the similarity score between two sets of features
    # Example: Use cosine similarity
    similarity_score = cosine_similarity(features1, features2)[0][0]
    return similarity_score
def bimodal_identification(fingerprint_image_path, finger_vein_image_path):
    # Preprocess fingerprint and finger vein images
    fingerprint_image = preprocess_fingerprint(cv2.imread(fingerprint_image_path))
    finger_vein_image = preprocess_finger_vein(cv2.imread(finger_vein_image_path))

    # Extract features using fingerprint and finger vein models
    fingerprint_features = fingerprint_model.predict(np.expand_dims(fingerprint_image, axis=0))
    finger_vein_features = finger_vein_model.predict(np.expand_dims(finger_vein_image, axis=0))

    # Calculate similarity score between features
    similarity_score = calculate_similarity(fingerprint_features, finger_vein_features)

    return similarity_score


# In[ ]:


threshold = 0.7  # Set a threshold for making a match/non-match decision
fingerprint_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/fp/1/1__M_Left_index_finger.BMP'
finger_vein_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/desiredfv/001/L_Fore/01.bmp'

similarity_score = bimodal_identification(fingerprint_image_path, finger_vein_image_path)

if similarity_score > threshold:
    print('Bimodal Biometric Match!')
else:
    print('Bimodal Biometric Non-Match')


# In[ ]:


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models for face and finger vein recognition
face_model = load_model('face_model.h5')  # Replace with your face recognition model
finger_vein_model = load_model('vein_model.h5')  # Replace with your finger vein model

def preprocess_image(image_path, target_size=(100, 100), is_grayscale=False):
    # Load the image
    image = cv2.imread(image_path)
    
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
    else:
        print(f"Failed to load the image: {image_path}")
        return None

# Define a function to perform bimodal biometric identification
def bimodal_identification(face_image_path, finger_vein_image_path):
    # Preprocess face and finger vein images
    face_image = preprocess_image(face_image_path, target_size=(100, 100))
    finger_vein_image = preprocess_image(finger_vein_image_path, target_size=(100, 100), is_grayscale=True)

    if face_image is None or finger_vein_image is None:
        return None
    
    # Extract features using face and finger vein models
    face_features = face_model.predict(np.expand_dims(face_image, axis=0))
    finger_vein_features = finger_vein_model.predict(np.expand_dims(finger_vein_image, axis=0))

    # Calculate similarity score between features (e.g., cosine similarity)
    similarity_score = cosine_similarity(face_features, finger_vein_features)[0][0]

    return similarity_score

# Define a threshold for making a match/non-match decision
threshold = 0.7  # Adjust as needed
# Example usage:
face_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/facenew/Aaron_Patterson/Aaron_Patterson_0001.jpg'  # Replace with your face image path
finger_vein_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/desiredfv/001/L_Fore/01.bmp'  # Replace with your finger vein image path



similarity_score = bimodal_identification(face_image_path, finger_vein_image_path)

if similarity_score is not None:
    print(f'Similarity Score: {similarity_score}')
    
    if similarity_score > threshold:
        print('Bimodal Biometric Match!')
    else:
        print('Bimodal Biometric Non-Match')


# In[ ]:


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity  # You may need to install scikit-learn

# Load pre-trained models for each modality
fingerprint_model = load_model('fingerprint_model.h5')  # Replace with your fingerprint model
facial_model = load_model('face_model.h5')  # Replace with your facial recognition model

def preprocess_fingerprint(image):
    if image is None or image.size == 0:
        # Handle the case where the image loading fails or it's empty
        print("Warning: Failed to load or empty fingerprint image.")
        return None

    # Resize the fingerprint image to the desired size (e.g., 100x100)
    image = cv2.resize(image, (100, 100))

    # Normalize pixel values to the range [0, 1]
    image = image.astype(np.float32) / 255.0

    # Expand dimensions to match the model's input shape
    image = np.expand_dims(image, axis=-1)

    return image

def preprocess_facial(image):
    if image is None or image.size == 0:
        # Handle the case where the image loading fails or it's empty
        print("Warning: Failed to load or empty facial image.")
        return None

    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the facial image to the desired size (e.g., 100x100)
    gray_image = cv2.resize(gray_image, (100, 100))

    # Normalize pixel values to the range [0, 1]
    gray_image = gray_image.astype(np.float32) / 255.0

    # Expand dimensions to match the model's input shape
    gray_image = np.expand_dims(gray_image, axis=-1)

    return gray_image

def calculate_similarity(features1, features2):
    # Calculate the similarity score between two sets of features
    similarity_score = cosine_similarity(features1, features2)[0][0]
    return similarity_score

threshold = 0.7  # Set a threshold for making a match/non-match decision

# Paths to test images for both fingerprint and face
fingerprint_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/fp/1/1__M_Left_index_finger.BMP'
facial_image_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/facenew/Aaron_Patterson/Aaron_Patterson_0001.jpg'

# Preprocess fingerprint and facial images
fingerprint_image = preprocess_fingerprint(cv2.imread(fingerprint_image_path))
facial_image = preprocess_facial(cv2.imread(facial_image_path))

# Extract features using fingerprint and facial models
fingerprint_features = fingerprint_model.predict(np.expand_dims(fingerprint_image, axis=0))
facial_features = facial_model.predict(np.expand_dims(facial_image, axis=0))

# Calculate similarity score between features
similarity_score = calculate_similarity(fingerprint_features, facial_features)

# Make identity decision based on the similarity score
if similarity_score > threshold:
    print('Bimodal Biometric Match!')
else:
    print('Bimodal Biometric Non-Match')


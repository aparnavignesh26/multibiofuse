import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Path to the directory
dataset_path = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/desiredfv'


# Create an instance of ImageDataGenerator with pixel value rescaling
datagen = ImageDataGenerator(rescale=1./255)

# Generate image data batches from a directory
data_generator = datagen.flow_from_directory(
       dataset_path,    # Path to the directory containing images
        target_size=(224, 224),  # Resize images to 224x224 pixels
        batch_size=32,        # Number of images in each batch
        class_mode='binary',  # Binary classification problem (two classes)
        shuffle=False)        # images are not shuffled

# Initialize an empty list to store the preprocessed images
preprocessed_images = []

# Loop through each batch of images generated by the data generator
for batch_images in data_generator:
    # Extract the batch of images
    batch_images = batch_images[0]
    
    # Append the batch of images to the list
    preprocessed_images.append(batch_images)
    
    # Break the loop if all images have been processed
    if len(preprocessed_images) * data_generator.batch_size >= data_generator.samples:
        break

# Convert the list of preprocessed images to a NumPy array
preprocessed_dataset = np.vstack(preprocessed_images)

# Save the preprocessed dataset
np.save('preprocessed_dataset.npy', preprocessed_dataset)
import matplotlib.pyplot as plt

# Load the preprocessed dataset
preprocessed_dataset = np.load('preprocessed_dataset.npy')

# Choose a random index to visualize an image
random_index = np.random.randint(12, preprocessed_dataset.shape[0])

# Visualize the original and preprocessed images
original_image = preprocessed_dataset[random_index]  # The original image before preprocessing
preprocessed_image = preprocessed_dataset[random_index]  # The preprocessed image

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(preprocessed_image)
plt.title('Preprocessed Image')

plt.tight_layout()
plt.show()

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# Load the preprocessed dataset
preprocessed_dataset = np.load('preprocessed_dataset.npy')

# Calculate basic statistics of the dataset
num_images = preprocessed_dataset.shape[0]
image_shape = preprocessed_dataset.shape[1:]

print("Number of images:", num_images)
print("Image shape:", image_shape)

# Display a random sample of images
num_samples = 10
sample_indices = np.random.randint(0, num_images, num_samples)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    plt.subplot(1, num_samples, i+1)
    plt.imshow(preprocessed_dataset[idx], cmap='gray')
    plt.title(f"Image {idx}")
    plt.axis('off')
plt.show()

# Calculate and display the average pixel values of the dataset
average_pixel_values = np.mean(preprocessed_dataset, axis=(0, 1, 2))
print("Average pixel values:", average_pixel_values)

# Plot a histogram of pixel values
plt.figure(figsize=(10, 6))
plt.hist(preprocessed_dataset.flatten(), bins=50, color='c')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Pixel Value Distribution')
plt.show()

# Load the preprocessed dataset
preprocessed_dataset = np.load('preprocessed_dataset.npy')

def display_samples_with_grayscale(num_samples):
    # Choose random image indices to visualize
    random_indices = np.random.randint(0, preprocessed_dataset.shape[0], num_samples)

    plt.figure(figsize=(15, 8))
    for i, random_index in enumerate(random_indices):
        # Select the image to process
        image_to_process = preprocessed_dataset[random_index]

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2GRAY)

        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(image_to_process)
        plt.title(f'Original Image {random_index}')
        plt.axis('off')

        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Specify the number of samples to display
num_samples = 5

# Display the samples with grayscale images
display_samples_with_grayscale(num_samples)

from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import os

# Load the preprocessed dataset
preprocessed_dataset = np.load('preprocessed_dataset.npy')

# Initialize an empty list to store LBP images
lbp_images = []

# Calculate LBP features for each preprocessed image
for image in preprocessed_dataset:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_images.append(lbp_image)

# Convert the list of LBP images to a NumPy array
lbp_dataset = np.array(lbp_images)

# Save the LBP dataset
np.save('lbp_dataset.npy', lbp_dataset)
# Load the LBP dataset
lbp_dataset = np.load('lbp_dataset.npy')

# Choose an example LBP image to visualize (you can change the index)
example_image = lbp_dataset[0]

# Display the original LBP image
plt.figure(figsize=(6, 6))
plt.imshow(example_image, cmap='gray')
plt.title('Example LBP Image')
plt.axis('off')
plt.show()


import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the preprocessed dataset and labels
preprocessed_dataset = np.load('preprocessed_dataset.npy')
labels = np.load('lbp_dataset.npy')

# Reshape the labels to have the same dimensions as the images
reshaped_labels = labels.reshape(labels.shape[0], labels.shape[1], labels.shape[2], 1)

X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_dataset, reshaped_labels, test_size=0.2, random_state=42)

# Print the shapes to confirm
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
from tensorflow.keras.layers import Reshape

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(224 * 224, activation='sigmoid'),  # Adjust the output shape to match (batch_size, 224, 224, 1)
    Reshape((224, 224, 1))  # Reshape to match the desired output shape
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use 'binary_crossentropy' for binary classification
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
batch_size = 32
epochs = 10
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
# Print the training and validation accuracy
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

print("Training Accuracy:", train_accuracy[-1])
print("Validation Accuracy:", val_accuracy[-1])
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_resized, y_test_resized)

# Print the test accuracy
print("Test Accuracy:", test_accuracy)

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()

# Plot the training and validation accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

import os
import cv2
import numpy as np

# Define the path to the directory containing your image dataset
dataset_dir = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/fp'

# Create lists to store image data and labels 
image_data = []
labels = [] 

# Loop through the directory and load BMP image files
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.BMP'):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            if image is not None:
               
                image_data.append(image)
                labels.append(os.path.basename(root))  # Store the folder name as the label

# Convert image data and labels to NumPy arrays
image_data = np.array(image_data)
labels = np.array(labels)
# Perform exploratory analysis tasks
print(f"Total number of images: {len(image_data)}")

# Visualize a random sample of images from the dataset
sample_size = 5  # Change this to the number of images you want to visualize
sample_indices = np.random.choice(len(image_data), sample_size, replace=False)

for i, idx in enumerate(sample_indices):
    plt.subplot(1, sample_size, i + 1)
    plt.imshow(cv2.cvtColor(image_data[idx], cv2.COLOR_BGR2RGB))
    plt.title(f'Label: {labels[idx]}')
    plt.axis('off')

plt.show()


height, width, channels = image_data[0].shape
print(f"Image dimensions: Height={height}, Width={width}, Channels={channels}")

# Calculate some basic statistics about the images (e.g., mean and standard deviation)
image_means = np.mean(image_data, axis=(0, 1, 2))
image_stddevs = np.std(image_data, axis=(0, 1, 2))

print(f"Mean pixel values (R, G, B): {image_means}")
print(f"Standard deviation (R, G, B): {image_stddevs}")
fingerprint_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization for image enhancement
enhanced_image = cv2.equalizeHist(fingerprint_image)

# Display the original and enhanced images side by side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(fingerprint_image, cmap='gray')
plt.title('Original Fingerprint Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Fingerprint Image')
plt.axis('off')

plt.tight_layout()
plt.show()
fingerprint_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization for image enhancement
enhanced_image = cv2.equalizeHist(fingerprint_image)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(enhanced_image, 128, 255, cv2.THRESH_BINARY)

# Apply noise reduction using Gaussian blur
blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# Display the original, enhanced, binary, and blurred images
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(fingerprint_image, cmap='gray')
plt.title('Original Fingerprint Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Fingerprint Image')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image (Noise Reduction)')
plt.axis('off')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    image_data, labels, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Display the new shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Reshape y_train and y_test and convert to float32
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(103, 96, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,  
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming you have already trained your model and have predictions
y_pred = model.predict(X_test)

# Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5)
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype('int32')

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, average='micro')  # Use 'average' parameter here
recall = recall_score(y_test, y_pred_binary, average='micro')  # Use 'average' parameter here
f1 = f1_score(y_test, y_pred_binary, average='macro')  # Use 'average' parameter here

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Print the evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)


# Plot Accuracy-Loss curve (assuming you have 'history' available from model training)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')

plt.show()

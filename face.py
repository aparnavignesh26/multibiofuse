import os
import cv2
import numpy as np

# Path to the directory containing the LFW dataset
lfw_dir = 'C:/Users/Vignesh Sundararajan/Desktop/Project/Dataset/edited/facenew'
# Lists to store image data and labels
images = []
labels = []

# Loop through subdirectories (person names)
for person_name in os.listdir(lfw_dir):
    person_dir = os.path.join(lfw_dir, person_name)
    if os.path.isdir(person_dir):
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale image
            if image is not None:
                images.append(image)
                labels.append(person_name)

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Display some information about the dataset
print(f"Number of images: {X.shape[0]}")
print(f"Image shape: {X.shape[1:]}")
print(f"Number of unique labels: {len(np.unique(y))}")
import matplotlib.pyplot as plt
import seaborn as sns
# Display the distribution of labels
plt.figure(figsize=(25,15))
sns.histplot(y, kde=False)
plt.xticks(rotation=55)
plt.xlabel('Person')
plt.ylabel('Number of Images')
plt.title('Distribution of Images per Person')
plt.show()

# Display a few sample images
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f'Person: {y[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Preprocessing: Crop and Resize
def preprocess_image(image, target_size=(100, 100)):
    # Crop the center of the image
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    crop_size = min(image.shape[0], image.shape[1])
    cropped_image = image[
        center_y - crop_size // 2 : center_y + crop_size // 2,
        center_x - crop_size // 2 : center_x + crop_size // 2
    ]
    # Resize the image to the target size
    resized_image = cv2.resize(cropped_image, target_size)
    return resized_image

# Apply preprocessing to each image
X_preprocessed = np.array([preprocess_image(image) for image in X])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Reshape X_train and X_test to have an additional channel dimension
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)



# Ensure that the number of samples in X_train_reshaped and y_train_reshaped are equal
num_samples = min(X_train_reshaped.shape[0], y_train_reshaped.shape[0])
X_train_reshaped = X_train_reshaped[:num_samples]
y_train_reshaped = y_train_reshaped[:num_samples]

# Ensure that the number of samples in X_test_reshaped and y_test_reshaped are equal
num_samples = min(X_test_reshaped.shape[0], y_test_reshaped.shape[0])
X_test_reshaped = X_test_reshaped[:num_samples]
y_test_reshaped = y_test_reshaped[:num_samples]



# Display the new shapes
print("X_train_reshaped shape:", X_train_reshaped.shape)
print("X_test_reshaped shape:", X_test_reshaped.shape)
print("y_train_reshaped shape:", y_train_reshaped.shape)
print("y_test_reshaped shape:", y_test_reshaped.shape)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
])

# Compile the model with hyperparameters
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
batch_size = 32
epochs = 10

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_reshaped, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test_reshaped)
print("Test accuracy:", test_acc)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions on the test set
y_pred = model.predict(X_test_reshaped)

# Convert probabilities to binary labels (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate confusion matrix
confusion = confusion_matrix(y_test_reshaped, y_pred_binary)

# Generate a classification report
classification_rep = classification_report(y_test_reshaped, y_pred_binary)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(confusion)

print("\nClassification Report:")
print(classification_rep)

# Plot the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


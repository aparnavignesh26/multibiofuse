from google.colab import drive
drive.mount('/content/drive')
import os
data_dir = '/content/drive/MyDrive/MMCBNU_6000'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary

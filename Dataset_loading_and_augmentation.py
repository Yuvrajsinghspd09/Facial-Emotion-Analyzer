import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import random

# Set seeds for reproducibility
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# Define constants
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 80
NUM_CLASSES = 7

train_dir = "/kaggle/input/face-expression-recognition-dataset/images/train"
validation_dir = "/kaggle/input/face-expression-recognition-dataset/images/validation"

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

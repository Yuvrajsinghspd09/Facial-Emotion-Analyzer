# Facial-Emotion-Analyzer




# Facial Emotion Recognition Model

This project implements a convolutional neural network (CNN) for facial emotion recognition using TensorFlow and Keras. The model is trained to classify facial expressions into one of seven predefined emotion categories.

## Dataset

The dataset used for this project is the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset). It contains images of facial expressions categorized into seven classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Code Overview

1. **Imports and Setup**:
   - Import necessary libraries.
   - Set random seeds for reproducibility.
   - Define constants including image size, batch size, number of epochs, and number of classes.

2. **Data Preparation**:
   - Create data generators for training and validation datasets.
   - Apply data augmentation techniques such as rotation, width shift, height shift, shear, zoom, and horizontal flip.

3. **Model Creation**:
   - Define a CNN model with convolutional layers, batch normalization, max pooling, and dense layers.
   - Compile the model with the Adam optimizer and categorical crossentropy loss function.

4. **Model Training**:
   - Train the model using the training and validation data generators.
   - Measure and print the training time.

5. **Model Saving**:
   - Save the trained model to a file named `facial_emotion_model.h5`.

6. **Plotting Training History**:
   - Plot training and validation accuracy and loss over epochs.



Requirements:

TensorFlow
Numpy
Matplotlib


You can install the required libraries using :
pip install tensorflow numpy matplotlib


Usage:

Prepare Your Environment:
Ensure you have the necessary libraries installed.
Download the dataset and adjust the train_dir and validation_dir paths if necessary.

Run the Code:
Execute the script to train the model and generate the plots.
The trained model will be saved as facial_emotion_model.h5.

Analyze Results:
Review the training and validation accuracy and loss plots to assess model performance.


Acknowledgements
The dataset used in this project is provided by Kaggle.
Dataset Link - https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

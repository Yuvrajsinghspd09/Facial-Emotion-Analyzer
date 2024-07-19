import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('facial_emotion_model.h5')

# Dictionary to label all emotions
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

st.title("Facial Emotion Detection")

# Option to select between uploading an image or using the webcam
option = st.selectbox("Choose input method", ("Upload an Image", "Use Webcam"))

def preprocess_image(image):
    # Convert to RGB
    image = image.convert('RGB')
   
    image = image.resize((48, 48))
   
    img_array = np.array(image)
   
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if option == "Upload an Image":
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
       
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Preprocess the image for the model
        img_array = preprocess_image(image)
        # Make a prediction
        prediction = model.predict(img_array)
        emotion = np.argmax(prediction)
        # Display the prediction
        st.write(f'The detected emotion is: {emotion_dict[emotion]}')
        st.write(prediction)  # Debugging line to see the raw prediction values

elif option == "Use Webcam":
    # Capture image from webcam
    picture = st.camera_input("Take a picture")

    if picture:
        # Convert the image to PIL Image
        img_pil = Image.open(picture)
        # Display the captured image
        st.image(img_pil, caption="Captured Image", use_column_width=True)
        
        img_array = preprocess_image(img_pil)
        
        prediction = model.predict(img_array)
        emotion = np.argmax(prediction)
        # Display the prediction
        st.write(f'The detected emotion is: {emotion_dict[emotion]}')
        st.write(prediction)  

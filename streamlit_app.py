import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('test1.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])

def preprocess_image(image):
    img = image.resize((256, 256))  # Adjust the size to match your model's input
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        class_index = 0
    else:
        class_index = 1

    if class_index == 0:
        return 'Ambulance'
    else:
        return 'Vehicle'

def main():
    st.title("Ambulance and Vehicle Image Classification")
    st.text("Upload an image and the model will predict whether it's an ambulance or a vehicle.")

    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary directory
        image = Image.open(uploaded_file)
        result = classify_image(image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Show the prediction
        st.write(f"Prediction: {result}")

if __name__ == '__main__':
    main()

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
    confidence = prediction[0] * 100
    if prediction[0] > 0.5:
        class_index = 0
    else:
        class_index = 1

    if class_index == 0:
        return 'Ambulance', confidence
    else:
        return 'Vehicle', confidence

def main():
    st.title("Ambulance and Vehicle Image Classification")
    st.text("Upload an image and the model will predict whether it's an ambulance or a vehicle.")

    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary directory
        image = Image.open(uploaded_file)
        result, confidence = classify_image(image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Show the prediction and confidence
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}%")

if __name__ == '__main__':
    main()

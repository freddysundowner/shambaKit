import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# Load your .h5 model
model = load_model('best_model.h5')

# Define the class labels
class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy',
    38: 'Angular_leaf_spot',
    39: 'bean_healthy',
    40: 'bean_rust',
}

# Define the prediction function
def prediction(img):
    img = img.resize((256, 256), Image.Resampling.LANCZOS)  # Resize the image to match the model's expected input shape
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    pred = model.predict(img)
    return pred
# def prediction(img):
#     i = img_to_array(img)
#     im = preprocess_input(i)
#     img = np.expand_dims(im, axis=0)
#     pred = model.predict(img)
#     return pred

# Set the page background color
page_bg_img = '''
<style>
body {
background-color: #d4f5dc;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app
st.title('Plant Disease Classification App')
st.write('Upload an image or take a picture of a plant leaf to predict its disease.')

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
captured_image = st.camera_input("Take a picture...")

if uploaded_file or captured_image:
    if uploaded_file:
        img = Image.open(uploaded_file)
    else:
        img = Image.open(captured_image)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Just a second...")

    # Make prediction when button is clicked
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            pred = prediction(img)
            pred_class = np.argmax(pred, axis=1)[0]
            st.success(f'Prediction: {class_labels[pred_class]}')


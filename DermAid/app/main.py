import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import time
from datetime import date
from fpdf import FPDF

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/skin_disease_prediction_model.h5"

# Custom Keras classes/functions
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super(CustomInputLayer, self).__init__(*args, **kwargs)

class CustomAdam(tf.keras.optimizers.Adam):
    def __init__(self, *args, **kwargs):
        if 'weight_decay' in kwargs:
            kwargs.pop('weight_decay')
        super(CustomAdam, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(CustomAdam, self).get_config()
        config.pop('weight_decay', None)
        return config

def get_custom_objects():
    return {
        'DTypePolicy': tf.keras.mixed_precision.Policy('float32'),
        'InputLayer': CustomInputLayer,
        'Adam': CustomAdam,
    }

custom_objects = get_custom_objects()

# Load model
try:
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam')
except FileNotFoundError:
    st.error("Model file not found. Please ensure that the model file exists in the specified path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load class indices
try:
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))
except FileNotFoundError:
    st.error("Class indices file not found. Please ensure the JSON file is present in the specified path.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error parsing class indices JSON file. Please check its contents.")
    st.stop()
except Exception as e:
    st.error(f"Error loading class indices: {str(e)}")
    st.stop()

def load_and_preprocess_image(image, target_size=(224, 224)):
    try:
        img = Image.open(image)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    if preprocessed_img is None:
        return None

    try:
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown class")
        return predicted_class_name
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Streamlit app UI
st.set_page_config(page_title="Skin Disease Identification", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    /* Remove all padding and margin */
   .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
   .main-container {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-top: -80px; /* Adjust this value to ensure the header is at the top */
    }
   .stButton > button {
        background-color: #3498db;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
    }
   .stButton > button:hover {
        background-color: #2980b9;
    }
   .prediction-result {
        font-size: 1.5rem;
        color: #27ae60;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title('🩺 Skin Disease Identification 🩺')
st.write("Please enter your details and upload an image of your skin to identify potential diseases using our AI model.")

# Patient Details Input
patient_name = st.text_input("Patient Name")

# Age input: Years and Months
col1, col2 = st.columns(2)

with col1:
    patient_age = st.number_input("Patient Age", min_value=0, max_value=150, value=0, step=1)

with col2:
    age_unit = st.selectbox("Age Unit", ["Years", "Months"], index=0)

patient_mobile_number = st.text_input("Patient Mobile Number")

today_date = date.today()
st.write(f"Today's Date: {today_date.strftime('%d-%m-%Y')}")

duration_list = ["Select", "1-3 days", "1-2 weeks", "1 month", "more than one month"]
duration = st.selectbox("How long has this been?", duration_list)

itching_list = ["Select", "Yes", "No"]
itching = st.selectbox("Is there itching?", itching_list)

body_parts_list = ["Select", "Face", "Arms", "Legs", "Torso", "Other"]
body_parts = st.selectbox("On which parts of the body?", body_parts_list)

family_symptoms_list = ["Select", "Yes", "No"]
family_symptoms = st.selectbox("Does anyone in your family have the same symptoms?", family_symptoms_list)

uploaded_image = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Check if all inputs are valid before making prediction
#...

if st.button("Submit"):
    if uploaded_image is not None and patient_name and patient_age > 0 and \
            (len(patient_mobile_number) == 10 and patient_mobile_number.isdigit()) and \
            duration != "Select" and itching != "Select" and \
            body_parts != "Select" and family_symptoms != "Select":

        # Process the image and make prediction
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)  # Show the image
        with st.spinner('Analyzing... Please wait'):
            time.sleep(2)  # Simulate processing delay
            prediction = predict_image_class(model, uploaded_image, class_indices)
            if prediction:
                st.success('Analysis Complete!')
                st.markdown(f"<div class='prediction-result'>Prediction: {prediction}</div>", unsafe_allow_html=True)

                # Create the PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=15)
                pdf.cell(200, 10, txt="Patient Report", ln=True, align='C')
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True, align='L')
                pdf.cell(0, 10, txt=f"Patient Age: {patient_age} {age_unit}", ln=True, align='L')
                pdf.cell(0, 10, txt=f"Patient Mobile Number: {patient_mobile_number}", ln=True, align='L')
                pdf.cell(0, 10, txt=f"Today's Date: {today_date.strftime('%d-%m-%Y')}", ln=True, align='L')
                pdf.cell(0, 10, txt=f"How long has this been?: {duration}", ln=True, align='L')
                pdf.cell(0, 10, txt=f"Is there itching?: {itching}", ln=True, align='L')
                pdf.cell(0, 10, txt=f"On which parts of the body?: {body_parts}", ln=True, align='L')
                pdf.cell(0, 10, txt=f"Does anyone in your family have the same symptoms?: {family_symptoms}", ln=True, align='L')
                pdf.cell(0, 10, txt=f" Prediction: {prediction}", ln=True, align='L')

                # Save the image in a temporary path for the PDF
                image_filename = f"{patient_name}_image.png"
                image = Image.open(uploaded_image)
                image.save(image_filename)
                pdf.image(image_filename, x=10, y=pdf.get_y() + 10, w=100)

                # Save the PDF to a bytes buffer
                pdf_output = pdf.output(dest='S')
                pdf_buffer = bytes(pdf_output, 'latin-1')

                # Download button
                st.download_button("Download Patient Report", data=pdf_buffer, file_name=f"{patient_name}_report.pdf")

            else:
                st.error("Failed to make a prediction. Please try again with a different image.")
    else:
        st.error("Please fill out all fields correctly before submitting.")
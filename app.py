import os
import pickle
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load the trained models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Define the class labels for lung disease detection
class_labels = {0: 'Healthy', 1: 'Type 1 disease', 2: 'Type 2 disease'}
lung_disease_model = tf.keras.models.load_model(f'{working_dir}/saved_models/model.h5')

# Function to make predictions for lung disease detection
@tf.function
def predict_lung_disease(image):
    result = lung_disease_model(image)
    label_index = tf.argmax(result, axis=1)
    return label_index

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Health Assistant Menu',
                           ['Diabetes Prediction ',
                            'Heart Disease Prediction',
                            "Parkinson's Prediction",
                            'Lung Disease Detection'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'lungs'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # Page title
    st.title('Diabetes Prediction using ML')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI value')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        Age = st.text_input('Age of the Person')

    # Prediction button
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
        st.success(diab_diagnosis)
        st.balloons()

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    # Page title
    st.title('Heart Disease Prediction using ML')

    # Input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
        restecg = st.text_input('Resting Electrocardiographic results')
        exang = st.text_input('Exercise Induced Angina')
    with col2:
        sex = st.text_input('Sex')
        thalach = st.text_input('Maximum Heart Rate achieved')
        oldpeak = st.text_input('ST depression induced by exercise')
    with col3:
        cp = st.text_input('Chest Pain types')
        slope = st.text_input('Slope of the peak exercise ST segment')
        ca = st.text_input('Major vessels colored by flourosopy')
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # Prediction button
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        st.success(heart_diagnosis)
        st.balloons()

# Parkinson's Prediction Page
elif selected == "Parkinson's Prediction":
    # Page title
    st.title("Parkinson's Disease Prediction using ML")

    # Input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        RAP = st.text_input('MDVP:RAP')
        APQ3 = st.text_input('Shimmer:APQ3')
        NHR = st.text_input('NHR')
        RPDE = st.text_input('RPDE')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        PPQ = st.text_input('MDVP:PPQ')
        APQ5 = st.text_input('Shimmer:APQ5')
        HNR = st.text_input('HNR')
        DFA = st.text_input('DFA')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        DDP = st.text_input('Jitter:DDP')
        APQ = st.text_input('MDVP:APQ')
        spread1 = st.text_input('spread1')
        spread2 = st.text_input('spread2')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        Shimmer = st.text_input('MDVP:Shimmer')
        DDA = st.text_input('Shimmer:DDA')
        D2 = st.text_input('D2')
        PPE = st.text_input('PPE')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    # Prediction button
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input = [float(x) for x in user_input]
        parkinsons_prediction = parkinsons_model.predict([user_input])
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
        st.success(parkinsons_diagnosis)
        st.balloons()

# Lung Disease Detection Page
elif selected == 'Lung Disease Detection':
    # Page title
    st.markdown("<h1 style='color: yellow; font-family: Arial, sans-serif; font-size: 24px;'>Lung disease Detector App by CodeWudaya 👌</h1>", unsafe_allow_html=True)
    st.write("Upload an image and the app will predict its class :🫁")

    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')  # Open the image using PIL
        image = image.resize((64, 64))  # Resize the image to the desired size
        image = np.array(image)  # Convert the image to a NumPy array
        image = np.expand_dims(image, axis=0)  # Add an extra dimension

        result = predict_lung_disease(image)
        prediction_index = result.numpy()[0]
        prediction_label = class_labels[prediction_index]

        if prediction_label == 'Healthy':
            prediction_color = '#00FF00'  # Green color
        else:
            prediction_color = '#FF0000'  # Red color
        st.markdown(f"<p style='font-size:24px;font-weight:bold;color:{prediction_color};'>{prediction_label}</p>", unsafe_allow_html=True)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.balloons()

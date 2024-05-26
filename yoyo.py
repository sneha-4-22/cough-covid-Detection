import streamlit.components.v1 as components
import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

model = load_model('covid.h5')

def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=5)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feature_list = []
    feature_list.append(np.mean(chroma_stft))
    feature_list.append(np.mean(rmse))
    feature_list.append(np.mean(spec_cent))
    feature_list.append(np.mean(spec_bw))
    feature_list.append(np.mean(rolloff))
    feature_list.append(np.mean(zcr))
    for e in mfcc:
        feature_list.append(np.mean(e))

    return feature_list[:19]

def preprocess_features(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(np.array(features).reshape(1, -1))
    return features_scaled

def generate_report(user_info, prediction, features):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "COVID-19 Detection Report")
    p.drawString(100, 730, f"Name: {user_info['name']}")
    p.drawString(100, 710, f"Age: {user_info['age']}")
    p.drawString(100, 690, f"Gender: {user_info['gender']}")
    p.drawString(100, 670, f"Prediction: {'COVID-19' if prediction[0] >= 0.9 else 'Non-COVID-19'}")

    fig, ax = plt.subplots()
    feature_names = ['Chroma STFT', 'RMSE', 'Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff', 'Zero Crossing Rate'] + [f'MFCC{i}' for i in range(1, 14)]
    sns.barplot(x=feature_names, y=features, ax=ax)
    plt.xticks(rotation=90)
    plt.title('Extracted Features')
    plt.tight_layout()
    fig.savefig("features.png")

    p.drawImage("features.png", 100, 400, width=400, height=200)
    p.save()
    buffer.seek(0)
    return buffer


tour_script = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/shepherd.js@8.0.0/dist/css/shepherd.css">
<style>
  body {
    background-color: #fffaf0 ;
  }
  .shepherd-element {
    border-radius: 10px;
    background-color: #fff;
    padding: 2px;
    max-width: 270px;
    background-color: #fffaf0; 
    border: 2px solid red;
    position: fixed;
    top: 50%;
    right: 20px;
    transform: translateY(-50%);
  }
  .shepherd-header {
    font-size: 1.2em;
    margin-bottom: 3px;
    background-color: #fffaf0; 
    color: #333;
    padding: 0px;
  }
  .shepherd-text {
    font-size: 1em;
    color: #555;
    background-color: #fffaf0; 
    max-height:350px;
    padding: 5px;
    border-radius: 10px;
  }
  .shepherd-button {
    background-color: #ff69b4; 
    color: #fff;
    border: none;
    padding: 5px 10px;
    border-radius: 5px;
    cursor: pointer;
    margin-top: 2px;
  }
  .shepherd-button:hover {
    background-color: #ff1493;
  }
  .shepherd-cancel-icon {
    color: red;
  }
  iframe {
    background-color:  #ff1493;
  }
  .shepherd-step .shepherd-content {
    padding: 0 !important;
  }
  .shepherd-step {
    margin: 0 !important;
    padding: 0 !important;
  }
  .shepherd-content {
    margin: 0 !important;
    padding: 0 !important;
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/shepherd.js@8.0.0/dist/js/shepherd.min.js"></script>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    const tour = new Shepherd.Tour({
      useModalOverlay: true,
      defaultStepOptions: {
        cancelIcon: { enabled: true, classes: 'shepherd-cancel-icon' },
        classes: 'shepherd-theme-arrows',
        scrollTo: { behavior: 'smooth', block: 'center' }
      }
    });

    tour.addStep({
      id: 'welcome',
      text: 'Welcome to the COVID-19 Detection App! This tour will guide you through the steps to use this app.',
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'name',
      text: 'First, enter your name.',
      attachTo: { element: '[data-testid="stTextInput"]', on: 'bottom' },
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'age',
      text: 'Next, enter your age.',
      attachTo: { element: '[data-testid="stNumberInput"]', on: 'bottom' },
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'gender',
      text: 'Select your gender.',
      attachTo: { element: '[data-testid="stRadio"]', on: 'bottom' },
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'upload',
      text: 'Upload the audio file of your cough.',
      attachTo: { element: '[data-testid="stFileUploader"]', on: 'bottom' },
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'submit',
      text: 'Click Submit to analyze the sound.',
      attachTo: { element: '[data-testid="stButton"]', on: 'bottom' },
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'result',
      text: 'The result will be displayed here based on the analysis. You can also download the report.',
      attachTo: { element: '.stMarkdown', on: 'top' },
      buttons: [{ text: 'Finish', action: tour.complete, classes: 'shepherd-button' }]
    });

    tour.start();
  });
</script>
"""

st.title("COVID-19 Detection App")
components.html(tour_script)
st.header("Enter your details")
name = st.text_input("First Name")
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.radio("Gender", ["Male", "Female", "Other"])


st.header("Upload the audio file")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button('Submit'):
        features = extract_features(uploaded_file)
        features_scaled = preprocess_features(features)
        prediction = model.predict(features_scaled)
        
        if prediction[0] >= 0.9:
            st.success("Prediction: COVID-19")
        else:
            st.success("Prediction: Non-COVID-19")

        st.header("Extracted Features")
        feature_names = ['Chroma STFT', 'RMSE', 'Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff', 'Zero Crossing Rate'] + [f'MFCC{i}' for i in range(1, 14)]
        fig, ax = plt.subplots()
        sns.barplot(x=feature_names, y=features, ax=ax)
        plt.xticks(rotation=90)
        plt.title('Extracted Features')
        plt.tight_layout()
        st.pyplot(fig)

        user_info = {
            'name': name,
            'age': age,
            'gender': gender
        }
        buffer = generate_report(user_info, prediction, features)
        st.download_button(label="Download Report", data=buffer, file_name="COVID-19_Report.pdf", mime="application/pdf")

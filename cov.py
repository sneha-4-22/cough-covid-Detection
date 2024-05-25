import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import streamlit.components.v1 as components

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

st.title("COVID-19 Detection from Cough Sounds")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"], key='uploader')

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button('Submit', key='submit_button'):
        features = extract_features(uploaded_file)
        features_scaled = preprocess_features(features)
        prediction = model.predict(features_scaled)
        if prediction[0] >= 0.9:
            st.write("Prediction: COVID-19")
        else:
            st.write("Prediction: Non-COVID-19")

tour_script = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/shepherd.js@8.0.0/dist/css/shepherd.css">
<style>
  .shepherd-element {
    border-radius: 10px;
    background-color: #fff;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    padding: 8px;
    max-width: 250px;
    border: 2px solid red;
  }
  .shepherd-header {
    font-size: 1.2em;
    margin-bottom: 3px;
    color: #333;
    padding: 0px;
    border-radius: 10px 10px 0 0;
  }
  .shepherd-text {
    font-size: 1em;
    color: #555;
    background-color: #fffaf0; 
    max-height:350px;
    padding: 5px;
    border-radius: 0 0 10px 10px;
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
      id: 'upload',
      text: 'First, upload a WAV file of a cough sound.',
      attachTo: { element: '[data-testid="stFileUploader"]', on: 'bottom' },
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'submit',
      text: 'After uploading the file, click the Submit button to analyze the sound.',
      attachTo: { element: '[data-testid="stButton"]', on: 'bottom' },
      buttons: [{ text: 'Next', action: tour.next, classes: 'shepherd-button' }]
    });

    tour.addStep({
      id: 'result',
      text: 'The result will be displayed here based on the analysis.',
      attachTo: { element: '.stMarkdown', on: 'top' },
      buttons: [{ text: 'Finish', action: tour.complete, classes: 'shepherd-button' }]
    });

    tour.start();
  });
</script>
"""

components.html(tour_script, height=230)

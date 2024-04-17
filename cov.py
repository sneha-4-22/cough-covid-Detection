import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

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

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button('Submit'):
        features = extract_features(uploaded_file)
        features_scaled = preprocess_features(features)
        prediction = model.predict(features_scaled)
        if prediction[0] >= 0.9:
            st.write("Prediction: COVID-19")
        else:
            st.write("Prediction: Non-COVID-19")

import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import preproces, CoughNet

# Load the trained model, scaler, and encoder
checkpoint = torch.load('checkpoints/checkpoint.pth')
model_state = checkpoint['model_state']
scaler = checkpoint['scaler']
encoder = checkpoint['encoder']
hparams = checkpoint['hparams']

model = CoughNet(len(hparams['features']))
model.load_state_dict(model_state)
model.eval()

# Define Streamlit app layout
st.title('Cough Detection App')

# File uploader widget
uploaded_file = st.file_uploader("Upload a WAV file", type="wav")

# Submit button
if st.button('Submit'):
    if uploaded_file is not None:
        # Perform inference
        df_features = pd.DataFrame(columns=hparams['features'])
        preprocessed_data = preproces(uploaded_file)
        feature_row = pd.DataFrame(preprocessed_data, index=[0])
        df_features = pd.concat([df_features, feature_row], ignore_index=True)
        X = np.array(df_features[hparams['features']], dtype=np.float32)
        X = torch.Tensor(scaler.transform(X))
        outputs = torch.softmax(model(X), 1)
        predictions = torch.argmax(outputs.data, 1)

        # Display results
        st.write(f"Model outputs: {outputs[0].detach().numpy()}")
        st.write(f"Predicted class: {encoder.classes_[predictions]}")

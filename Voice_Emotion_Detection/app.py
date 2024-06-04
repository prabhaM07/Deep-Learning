import streamlit as st
import numpy as np
import librosa
from keras.models import load_model

# Function to extract MFCC features
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Load the trained model
model = load_model("model.keras")

# Define emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

# Streamlit interface
st.title("Voice Emotion Detection")

uploaded_file = st.file_uploader("Upload a voice file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = f"temp_uploaded_file.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the uploaded voice file
    X = extract_mfcc(file_path)
    
    # Expand dimensions to match LSTM input: (batch_size, timesteps, features)
    X = np.expand_dims(X, axis=0)  # Adding batch dimension
    X = np.expand_dims(X, axis=-1)  # Adding features dimension

    # Make predictions
    predictions = model.predict(X)

    # Convert predictions to human-readable labels
    if predictions.shape[1] == len(emotion_labels):
        predicted_emotion = emotion_labels[np.argmax(predictions)]
        st.success(f"Predicted Emotion: {predicted_emotion}")
    else:
        st.error(f"Error: The model's output length {predictions.shape[1]} does not match the number of emotion labels {len(emotion_labels)}.")
else:
    st.info("Please upload a .wav file to get a prediction.")

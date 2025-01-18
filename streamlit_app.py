import streamlit as st
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("Model/the_best_model.keras")
#load saved tokenizor
tokenizer = joblib.load('Model/custom_tokenizer.joblib')

def model_prediction(X_test, model):
    y_pred = model.predict(X_test)
    Threshold = 0.4
    if (y_pred[0][0]>=Threshold):
        return "Formal";
    else:
        return "Informal";
#UI 
st.title("Check Your Text Here!....")
st.subheader("Formal or Informal Detector")
Input_text = st.text_input("Text ", placeholder= "Input Your Text Here...")

if (st.button('Predict Text Tone..')):
    try:
        if Input_text.strip() == "":
            st.error("Please enter some text before clicking the button.")
        else:
            text_sequences = tokenizer.texts_to_sequences([Input_text])
            if not text_sequences or not any(text_sequences[0]):
                raise ValueError("Tokenizer returned an empty sequence. Please check the input text.")
            text_padded = pad_sequences(text_sequences, maxlen=150, truncating='post', padding='post')
            with st.spinner("Predicting..."):
                prediction = model_prediction(text_padded, model)
                st.write(f"The text you provided is {prediction}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

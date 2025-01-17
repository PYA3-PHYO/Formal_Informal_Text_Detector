import streamlit as st
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('Model/best_model.h5')
#load saved tokenizor
tokenizer = joblib.load('Model/custom_tokenizer.joblib')

def model_prediction(X_test, model):
    y_pred = model.predict(X_test)
    threshold = 0.4
    if (y_pred[0][0]>=threshold):
        return "Formal";
    else:
        return "Informal";

#UI 
st.title("Check Your Text Here!....")
st.subheader("Formal or Informal Detector")

Input_text = st.text_input("Text ", placeholder= "Input Your Text Here...")
# st.button('Check the text tone..', on_click=click_button)

if st.button('Check the text tone..'):
    if Input_text.strip() == "":
        st.error("Please enter some text before clicking the button.")
    else:
        text_sequences = tokenizer.texts_to_sequences(Input_text) # Sequencing
        text_padded = pad_sequences(text_sequences,maxlen=150, truncating='post',padding='post') #padding
        prediction = model_prediction(text_padded, model)
        st.write(f"The text you provided is {prediction}.")
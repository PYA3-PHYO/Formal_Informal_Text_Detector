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
    value = y_pred[0][0]
    if (value>=Threshold):
        ans = "Formal"
        return ans;
    else:
        ans = "Informal"
        return ans;

#UI 
st.title("Check Your Text Here!....")
st.subheader("Formal or Informal Detector")
multi = '''This project is a Formal vs. Informal Text Detector that uses Natural Language Processing (NLP) techniques to classify the tone of a given text as either formal or informal. 

The system is built using a custom tokenizer and a deep learning model,(RNN, BiLSTM), trained to recognize language patterns typical of formal and informal writing. 

Users can input any text into the interface, and the model will analyze it, providing immediate feedback on the text's tone.
'''
st.markdown(multi)

st.subheader("Adjust Threshold Value")
st.write("Near 0 mean Informal and 1 for Formal. Adjust as you preferences, default is 0.4")
Threshold = st.slider("Value", 0.0, 1.0, 0.4)

Input_text = st.text_input("Text ", placeholder= "Input Your Text Here...")

if (st.button('Predict Text Tone..')):
    try:
        if Input_text.strip() == "":
            st.error("Please enter some text before clicking the button.")
        elif len(Input_text.split()) <= 1:
            st.error("Please enter more than one word.")
        else:
            text_sequences = tokenizer.texts_to_sequences([Input_text])
            text_padded = pad_sequences(text_sequences, maxlen=150, truncating='post', padding='post')
            with st.spinner("Predicting..."):
                prediction_ans, prediction_value = model_prediction(text_padded, model)
                prediction_value_round = round(prediction_value,2)
                st.write(f"The text you provided is *{prediction_ans} Text*..")
    except Exception as e:
        st.error(f"An error occurred: {e}")

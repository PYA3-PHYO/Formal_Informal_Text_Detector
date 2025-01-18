# Project Title: Formal and Informal Text Detector
## 1. Introduction
### 1.1 Project Overview: This project aims to develop a machine learning model capable of accurately classifying given text as either formal or informal. This classification can have various applications, assisting in content moderation for online platforms, and enhancing the quality of chatbots and conversational AI systems.

## 2. Methodology
### 2.1 Importing Necessary Packages
- pandas: For data manipulation and analysis.
- NumPy: For numerical operations and array manipulation.
- TensorFlow: For building and training the deep learning model.
- scikit-learn: For data splitting, model evaluation, and other machine learning utilities.
- nltk: For natural language processing tasks such as lemmatization and tokenization.
- emoji: For removing emojis from the text data.
- seaborn: For creating informative and visually appealing statistical graphics.
- matplotlib: For general plotting and visualization.
- joblib: For efficient saving and loading of trained models.

### 2.2 Data Source and Understanding  
The training data for this project was taken from https://zenodo.org/records/8023142. The dataset is the collection reddit comments of 3000 samples same meaning but include both formal and informal text syles. So, the total dataset is 6000 samples mix of formal and informal texts.
To understand, extract some samples from dataset and checking the shape of the dataset etc.
### 2.3 Data Preprocessing:
- Firstly,the dataset was uploaded on google drive and mount with Colab notebook.
- As the dataset is without label, split two 2 new data fram formal and informal respectively. Then add labels 0 and 1 manually and combine again asa final dataset. 
#### Text Cleaning:
- Convert all text to lowercase.
- Remove URLs using regular expressions.
- Remove emojis using the emoji library. Because the data is from Reddit comment it include emojis and we need to remove it.
#### Text Normalization:
- Perform lemmatization using the WordNetLemmatizer from the nltk library to reduce words to their base forms.
- Reason:  Lemmatization is more accurate base forms that are real words, which helps in understanding the text better and more readable and meaningful, making it suitable for our project.
#### Tokenization:
- Tokenize the text into individual words using the Tokenizer from the tensorflow library with max_word of 3000 along with out of vocab(OOV).
#### Sequencing and Padding:
- Convert text sequences into numerical representations using the texts_to_sequences method.
- Pad sequences to a fixed length (max_length=150) using pad_sequences to ensure consistent input dimensions for the model.
### 2.4 Modelling:
- Model Choice: A Bidirectional LSTM model is chosen for this task  because LSTMs are well-suited for sequential data like text, as they can effectively capture long-range dependencies. Bidirectionality allows the model to process information in both forward and backward directions, enhancing its ability to understand the context of words within a sentence.

#### Model Architecture:
- Embedding Layer: Maps each word to a dense vector representation, capturing semantic relationships between words.
- Bidirectional LSTM Layer: Processes the input sequence in both forward and backward directions, capturing contextual information from both directions.
- Dense Layers:
   - A fully connected hidden layer with 64 units and ReLU activation.
   - A dropout layer with a rate of 0.5 is applied to prevent overfitting.
   - An output layer with a single unit and sigmoid activation produces the probability of the input text being formal.
- Training Process:
   - The model is trained using the Adam optimizer and binary crossentropy loss.
   - Early stopping is implemented to prevent overfitting by monitoring validation loss and stopping training when it starts to increase.
   - The model is trained for a maximum of 10 epochs with a batch size of 32.
### 2.4 Model Evaluation:
   - Accuracy: 89.16%.
   - Precision: 89.2%
   - Recall: 89.2%
   - F1-score: 89.16%
- Confusion Matrix:
   - A confusion matrix is generated to visualize the model's predictions and identify areas of misclassification
   - ![Screenshot 2025-01-18 141703](https://github.com/user-attachments/assets/a57f9bd6-92a3-4355-be23-7966c780abd4)

## 3. Streamlit
### 3.1 Overview
The project includes a user-friendly web application built using **Streamlit**, which allows users to input text and receive real-time predictions about whether the text is formal or informal. 
![Screenshot 2025-01-18 142029](https://github.com/user-attachments/assets/f9701ac2-7fbd-4ee3-bfc9-5352e21dd665)

### 3.2 User Interface
- **Input Field**: Users can enter any text they wish to analyze.
- **Button**: A button labeled "Check the text tone.." triggers the model to classify the input text as formal or informal.

### 3.3 Error Handling
The application includes error handling to ensure smooth user experience:
- **Empty Input**: If the user submits an empty input, an error message prompts them to enter text.
- **Single Word Input**: The application checks if the input consists of more than one word, and an appropriate message is displayed if not.

## 4. Running the Project
### 4.1 Project Resources
   - Dataset -> Dataset/Training data source.csv
   - Model Training Source code -> Model/Model_training_source_code.ipynb
   - Tokenizore -> Model/custom_tokenizer.joblib
   - Model -> Model/the_best_model.keras
   - Streamlit app -> streamlit_app.py
   - 
### 4.2 Prerequisites
- Python (version 3.6 or higher)
- pip 

 ```bash
   pip install -r requirements.txt
   streamlit run stream_app.py

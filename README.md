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

### 2.2 Data Understanding  
Data Source: The training data for this project was taken from https://zenodo.org/records/8023142. The dataset is the collection reddit comment fro 3000 samples same meaning but include both formal and informal syles. So, the total dataset is 6000 samples mix of formal and informal texts.
Extract some samples from dataset and checking the shape of the dataset etc.
### 2.3 Data Preprocessing:
Firstly, dataset was uploaded on google drive and mount with Colab notebook.
As the dataset is without label, split two 2 new data fram formal and informal respectively. Then add labels 0 and 1 manually and combine again asa final dataset. 
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
## 2.4 Model Evaluation:
   - Accuracy: 89.16%.
   - Precision: 89.2%
   - Recall: 89.2%
   - F1-score: 89.16%
- Confusion Matrix:
   - A confusion matrix is generated to visualize the model's predictions and identify areas of misclassification


# 🧠 Next Word Prediction App (LSTM)

A deep learning-based Next Word Prediction system built using LSTM (Long Short-Term Memory) networks in TensorFlow and deployed with Streamlit.

This app takes a user input sentence and predicts the top 3 most probable next words, helping simulate real-world NLP applications like autocomplete and text generation.

---

## 🚀 Live Demo

👉 https://nextwordpredication-5td56evexrvyx6paaervka.streamlit.app/

---

## 📌 Project Overview

This project focuses on building a Natural Language Processing (NLP) model that learns patterns from text data and predicts the next word in a sequence.

**Key Features:**
- Predicts Top 3 Next Words
- Powered by LSTM Neural Network
- Real-time predictions using Streamlit UI
- Pre-trained model loaded for fast inference
- Handles unknown inputs gracefully

---

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pickle (for saving tokenizer)

---

## 🧠 Model Architecture

The model is built using a Sequential deep learning architecture:

* **Embedding Layer**
   - Input dimension: 63 (vocabulary size)
   - Output dimension: 100
* **LSTM Layer**
   - 150 units
* **Dense Layer**
   - Softmax activation for multi-class classification

---

## 🔄 Workflow

**1. Data Preprocessing**
- Load text dataset
- Tokenize text using Keras ```Tokenizer```
- Convert sentences into sequences
- Generate n-gram sequences
- Apply padding to ensure uniform input size

**2. Feature Engineering**
- Split into:
  - **X (input sequences)**
  - **y (target word)**
- Convert target into categorical format

**3. Model Training**
- Loss: ```categorical_crossentropy```
- Optimizer: ```adam```
- Epochs: up to 150 (with EarlyStopping)
- Validation split: 20%

**4. Model Saving**
- ```.h5``` → trained model
- ```.pkl``` → tokenizer
- ```.txt``` → max sequence length

---

## 💡 How Prediction Works
**1.** User enters a sentence  
**2.** Text is cleaned & tokenized  
**3.** Sequence is padded to match model input  
**4.** Model predicts probability distribution  
**5.** Top 3 highest probability words are returned

You can also watch the video below for a quick demo:

---

# 📦 Installation & Setup

## 1. Download the repository

It will be downloaded in Zip file so you have to **Extract** the files. A direct link to the repository is available below or click on given **link** and you will be redirected to the repository.

[https://github.com/rohitvirdi5rv-crypto/Next_Word_Predication.git](https://github.com/rohitvirdi5rv-crypto/Next_Word_Predication.git)

---
## 2. Create a virtual environment

```
python -m venv venv
```
---
## 3. Activate Environment

Activate it on Windows:
```
venv\Scripts\activate
```
---

## 4. Install Dependencies
```
pip install -r requirements.txt
```
---
## 5. Run the Streamlit app

```
streamlit run app.py
```

The application will open in your browser.

---

## 🔥 Future Improvements
- Add larger dataset for better accuracy
- Use Bidirectional LSTM / GRU
- Deploy using Docker
- Add autocomplete typing feature
- Improve UI with suggestions dropdown

---

## 👨‍💻 Author

**Rohit Virdi**  
BCA + MCA Graduate  
Aspiring **Data Analyst / Data Scientist**  
Skilled in **Python, Numpy, Pandas, SQL, Data Analysis, Machine Learning, Deep Learning, NLP and Visualization**  
Interested in building data-driven solutions and intelligent systems

---
## ⭐ Support

If you like this project:

- ⭐ Star the repo
- 🍴 Fork it
- 🧠 Try improving the model

---

🔗 GitHub: https://github.com/rohitvirdi5rv-crypto

---
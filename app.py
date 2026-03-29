import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# --- 1. Load the Model and Tokenizer ---

# Load the saved Keras model
try:
    model = tf.keras.models.load_model("Next_Word_Predictor.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the saved Tokenizer
try:
    with open('tokenizer.pkl', 'rb') as handle:
        token = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Load max_len
try:
    with open('max_len.txt', 'r') as f:
        MAX_SEQUENCE_LEN = int(f.read().strip())
except Exception as e:
    st.error(f"Error loading max_len: {e}. Please ensure 'max_len.txt' exists.")
    st.stop()

# Global variable for word_index (for efficiency)
WORD_INDEX = token.word_index
INDEX_TO_WORD = {index: word for word, index in WORD_INDEX.items()}


# --- 2. Prediction Function ---

def predict_next_words(seed_text, num_predictions=3):
    """
    Predicts the top N next words for a given seed text.
    """
    # 1. Tokenize the seed text
    token_word = token.texts_to_sequences([seed_text])[0]
    
    # Check if input has known words
    if not token_word:
        return []

    # 2. Pad the sequence
    # Note: Use MAX_SEQUENCE_LEN - 1 because your model's input_length is 11, 
    # but the input sequence X was created as pre_padded_sequence[:,:-1], 
    # meaning its length is max_len - 1 (12-1 = 11 in your case, but let's be safe).
    # We must ensure the input size matches the model's input_length (11).
    padded_token_word = pad_sequences(
        [token_word], 
        maxlen=MAX_SEQUENCE_LEN - 1, 
        padding='pre'
    )

    # 3. Predict probabilities
    predictions = model.predict(padded_token_word, verbose=0)[0]

    # 4. Get the indices of the top N words
    # - np.argsort gives indices that would sort the array
    # - [::-1] reverses the array to get descending order
    # - [:num_predictions] takes the top N indices
    top_indices = np.argsort(predictions)[::-1][:num_predictions]

    # 5. Map indices back to words
    predicted_words = [INDEX_TO_WORD.get(index) for index in top_indices if index in INDEX_TO_WORD]
    
    return predicted_words


# --- 3. Streamlit UI ---

def main():
    st.set_page_config(page_title="Next Word Predictor", layout="centered")

    st.title("🧠 Next Word Prediction App")
    st.markdown("""
        Enter a starting sentence, and I will predict the **top 3 possible next words** using a pre-trained Keras LSTM model.
        
        The model's input length is **11** words.
    """)
    st.markdown("---")

    # Input Field
    seed_text = st.text_input(
        "Enter your sentence:",
        placeholder="e.g., The quick brown fox jumps",
        key="input_text"
    )

    # Prediction Button
    if st.button("Predict Next Words", help="Click to generate predictions"):
        if not seed_text or seed_text.isspace():
            st.warning("Please enter some text to start the prediction.")
            return

        with st.spinner('Thinking...'):
            # Clean the input (optional, but good practice)
            cleaned_text = seed_text.lower().strip()
            
            # Predict the top 3 words
            predicted_words = predict_next_words(cleaned_text, num_predictions=3)

        st.markdown("### Top 3 Predicted Next Words")

        if predicted_words:
            # Display the predictions in a clear, column format
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"**1st:** {predicted_words[0]}")
            
            if len(predicted_words) > 1:
                with col2:
                    st.info(f"**2nd:** {predicted_words[1]}")
            
            if len(predicted_words) > 2:
                with col3:
                    st.info(f"**3rd:** {predicted_words[2]}")
                    
            st.markdown("---")
            
            # Show a suggested next sentence
            suggested_sentence = f"{seed_text} **{predicted_words[0]}**"
            st.markdown(f"**Suggested continuation:** *{suggested_sentence}*")
            
        else:
            st.error("Could not generate a prediction. This might be because the input contains words the model has not seen before.")
            
if __name__ == '__main__':
    main()
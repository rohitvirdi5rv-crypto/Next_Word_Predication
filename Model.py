import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

token = Tokenizer()

file_path = r"D:\Revision\Projects\Next_Word_Predictor\next_word_prediction_10000_lines.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

token.fit_on_texts([data])

sequence = []
for sentence in data.split('\n'):
    print(sentence)
    tokenized_sentence = token.texts_to_sequences([sentence])[0]
    print(tokenized_sentence)
    for i in range(1,len(tokenized_sentence)):
        sequence.append(tokenized_sentence[:i+1])

max_len = max([len(x) for x in sequence])

pre_padded_sequence = pad_sequences(sequence, maxlen = max_len, padding = 'pre')

X = pre_padded_sequence[:,:-1]
y = pre_padded_sequence[:,-1]

y = to_categorical(y,num_classes = 63)

model = Sequential()
model.add(Embedding(63, 100, input_length=11))
model.add(LSTM(150))
model.add(Dense(63, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.build(input_shape=(None, 11))

early_stop = EarlyStopping(monitor='loss', patience=5)

model.fit(X, y, epochs=150, validation_split=0.2, callbacks=[early_stop])

words = "The user evaluates language"

for i in range(3):

    token_word = token.texts_to_sequences([words])[0]

    padded_token_word = pad_sequences([token_word], maxlen = 11, padding='pre')
    pos = np.argmax(model.predict(padded_token_word))

    for text,index in token.word_index.items():
        if  index == pos:
            print(text)

            text = text + " " + text
            print(text)

# Save the Keras model
model.save("Next_Word_Predictor.h5")

# Save the Tokenizer object
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the max_len for use in the Streamlit app
with open('max_len.txt', 'w') as f:
    f.write(str(max_len))

print("Model, Tokenizer, and max_len saved successfully.")

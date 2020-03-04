import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model

df = pd.read_csv('filename.csv')
df = df.drop(df.index[[5,6]])
txt = ''
count = 0
for art in df['article']:
    txt = txt+art
    
def preprocess(txt):
    pattern = re.compile(r'[^\u0621-\u064A]')
    txt = re.sub(pattern, ' ', txt)
    txt = re.sub(r'\s+',' ',txt)
    return txt

# learn for characters, each 50 characters will create a sequence
def create_seq(txt):
    num_of_records = len(txt)-100
    seq_len = 50
    X = []
    Y = []
    for i in range(num_of_records-seq_len):
        X.append(txt[i:i+seq_len+1])
        Y.append(txt[seq_len+i])
    return X, Y

def encode_sequeces(x):
    encoded_sequences = list()
    for sen in x:
        encoded = [stoi[char] for char in sen]
        encoded_sequences.append(encoded)
    return encoded_sequences

# preprocessing
cleaned_text = preprocess(txt)

# encoding characters to deal with intergers
unique_chars = sorted(list(set(cleaned_text)))
stoi = dict((c, i) for i, c in enumerate(unique_chars))
itos = dict((i, c) for i, c in enumerate(unique_chars))

# creating sequences
x, y = create_seq(cleaned_text)
encoded_sequences = encode_sequeces(x)

vocab_size = len(unique_chars)
seq_array = np.array(encoded_sequences)
X, y = seq_array[:,:-1], seq_array[:, -1]
y = to_categorical(y, num_classes=vocab_size)

train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=42)

# build model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=50, trainable=True))
model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

model.fit(train_x, train_y, epochs=10, verbose=2, validation_data=(val_x, val_y))

# saving model
model.save("pre_text_model_k.h5")
# loading model
model1 = load_model('pre_text_model_k.h5')

def predict_word(model1, in_text):
    for _ in range(15):
        encoded_intext = [stoi[char] for char in in_text]
        # to get an input of 50 characters
        encoded = pad_sequences([encoded_intext], maxlen=50, truncating='pre')
        yhat = model1.predict_classes(encoded, verbose=0)
        new_char = itos[yhat[0]]
        in_text=in_text+new_char
        if new_char == ' ':
            return in_text

        
# Example
print(predict_word(model1, 'بعد '))
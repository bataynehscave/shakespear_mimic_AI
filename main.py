import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Activation, Input
from tensorflow.keras.optimizers import RMSprop

file_path = './data/shakespeare.txt'

text = open(file_path, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c,i) for i, c in enumerate(characters))

index_to_char = dict((i,c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

'''
sentences = []
next_characters = []
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    
    y[i, char_to_index[next_characters[i]]] = 1

np.save('x_train', x)
np.save('y_train', y)
'''
model = tf.keras.models.load_model('shake_spear_gen_1.keras')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_idx = random.randint(0, len(text)-SEQ_LENGTH - 1)
    generated_text = ''
    sentence = text[start_idx: start_idx + SEQ_LENGTH]
    generated_text += sentence 
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1
        preds = model.predict(x, verbose=0)[0]
        next_idx = sample(preds, temperature)
        next_char = index_to_char[next_idx]
        generated_text += next_char
        sentence = sentence[1:] + next_char
    
    return generated_text

print(generate_text(1000, 1))


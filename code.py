#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:06:27 2019

This project is a study of sentiment ana,ysis using the data "Sentiment Labelled Sentences Data Set"
from Kaggle. The link to the dataset is below:
    https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set
    
It is a supervised learning task, with y being 1 (positive review) and 0 (negaive review).

Scores are on an integer scale from 1 to 5. We considered reviews with a score of 4 and 5 to be positive, and scores of 1 and 2 to be negative. 


Input x is a sentence, grabed from 3 websites by the auther:
    imdb.com, amazon.com and yelp.com

@author: yuxing
"""

import os

import numpy as np
import pandas as pd

from numpy.random import rand, randn

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# =============================================================================
#  set up working directory
# =============================================================================

data_dir0 = "/Users/yuxing/Documents/ML_data/"
code_dir0 = "/Users/yuxing/Documents/ML_code/Tensorflow/"

project_name = "Sentiment analysis/"

data_dir = code_dir0 + project_name
code_dir = code_dir0 + project_name

output_dir = code_dir

if os.getcwd() != code_dir:
    os.chdir(code_dir)

print("Current working directory is:" + os.getcwd())


# =============================================================================
# Assemble the data
# =============================================================================

with open(data_dir + "amazon_cells_labelled.txt") as f1:
    lines = f1.readlines()

with open(data_dir + "imdb_labelled.txt") as f1:
    temp = f1.readlines()
    lines=lines+temp

with open(data_dir + "yelp_labelled.txt") as f1:
    temp = f1.readlines()
    lines=lines+temp
    
x = []
y = []
for value in lines:
    temp = value.split('\t')
    x.append(temp[0])
    temp[1].replace('\n','')
    y.append(int(temp[1]))
    
# =============================================================================
#  tokenize sentences and split them into train and test sets
# =============================================================================

    
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=2500,split=' ')
tokenizer.fit_on_texts(x)

#tokenizer.word_index

# now the sentences in x still have variable length, we need to pad them
# to the same length


from keras.preprocessing.sequence import pad_sequences

X = tokenizer.texts_to_sequences(x)
X = pad_sequences(X)

Y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)


# =============================================================================
# Build the model structure
# =============================================================================



from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


batch_size = 32

model = Sequential()
model.add(Embedding(2500, 128, input_length = X.shape[1]))
model.add(Dropout(0.4))
model.add(LSTM(300, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(1, activation="sigmoid"))

model.summary()


model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])


history = model.fit(x_train, y_train, 
          batch_size = batch_size, 
          epochs = 10,
          verbose = 2,
          validation_data=(x_test, y_test))



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


score = model.evaluate(x_test, y_test, verbose = 2, batch_size=batch_size)

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
    
f1.close()
    
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

tokenizer = Tokenizer(num_words= 2500, split=' ')
tokenizer.fit_on_texts(x)

#tokenizer.word_index

# now the sentences in x still have variable length, we need to pad them
# to the same length


from keras.preprocessing.sequence import pad_sequences

X = tokenizer.texts_to_sequences(x)

word_index = tokenizer.word_index

print("Found {} unique tokens.".format(len(word_index)))

X = pad_sequences(X)

Y = np.array(y)

print("Shape of X is", X.shape)
print("Shape of Y is", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)



# =============================================================================
# Use the existing embeddings downloaded from the website
# =============================================================================

# Download the glove.6B.zip from the website below

https://nlp.stanford.edu/projects/glove/


embeddings_index = {}
f = open(os.path.join(data_dir + "glove.6B/", 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# this "embedding index" maps words to a 100-D vector

Embedding_dim = len(embeddings_index['the'])


#we can leverage our embedding_index dictionary and our word_index to 
#compute our embedding matrix

embedding_matrix = np.zeros((len(word_index) + 1,  Embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        


    

# =============================================================================
# Build the model structure
# =============================================================================


from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalMaxPooling1D

from tensorflow.keras.models import Sequential, Model


batch_size = 16



# load static pre-trained embedding

embedding_layer = Embedding(len(word_index) + 1,
                            Embedding_dim,
                            weights=[embedding_matrix],
                            input_length = X.shape[1],
                            trainable = False)


#
#
#sequence_input = Input(shape = (X.shape[1],), dtype = "int32")
#embedded_sequences = embedding_layer(sequence_input)
#x = Dropout(0.2)(embedded_sequences)
#x = Bidirectional(LSTM(200, activation="relu", return_sequences = True))(x)
#x = Dropout(0.2)(x)
#x = Bidirectional(LSTM(200, activation="relu"))(x)
#x = Dropout(0.2)(x)
#Output = Dense(1, activation="sigmoid")(x)
#
#model = Model(sequence_input, Output)



# using an embedding layer to train the coeffcients

#model = Sequential()
#model.add(Embedding(2500, 128, input_length = X.shape[1]))
#model.add(Dropout(0.3))
##model.add(LSTM(300, activation="relu"))
#model.add(Bidirectional(LSTM(128, activation="sigmoid")))
#model.add(Dropout(0.3))
#model.add(Dense(1, activation="sigmoid"))


# Build a 1D convnet to solve the problem

sequence_input = Input(shape = (X.shape[1],), dtype = "int32")
embedded_sequences = embedding_layer(sequence_input)

#embedded_sequences = Embedding(2500, 128)(sequence_input)


x = Conv1D(256, 3, activation='relu')(embedded_sequences)
x = Conv1D(256, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
#x = Conv1D(128, 3, activation='relu')(x)
#x = MaxPooling1D(3, padding='same')(x)
#x = Conv1D(128, 5, activation='relu')(x)
#x = MaxPooling1D(35)(x)  # global max pooling
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)
#x = Flatten()(x)
x = Dense(32, activation='relu')(x)
Output = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, Output)

model.summary()


model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

#model.compile(loss = "sparse_categorical_crossentropy",
#              optimizer = "adam",
#              metrics = ["accuracy"])


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

#There's instability in the loss caluclation, due to the clipping of the
#sigmoid output from the last layer.
#https://stackoverflow.com/questions/52125924/why-does-sigmoid-crossentropy-of-keras-tensorflow-have-low-precision


score = model.evaluate(x_test, y_test, verbose = 2, batch_size=batch_size)

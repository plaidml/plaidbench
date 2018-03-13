#!/usr/bin/env python
import numpy as np

# Load the dataset and scrap everything but the first 16 entries
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 20000
max_length = 80

if __name__ == '__main__':
    print('Fetching the imdb dataset')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = x_train[:16]
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    np.save('imdb16.npy', x_train)

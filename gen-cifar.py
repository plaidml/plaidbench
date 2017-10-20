#!/usr/bin/env python
import numpy as np

# Load the dataset and scrap everything but the training images
# cifar10 data is too small, but we can upscale
from keras.datasets import cifar10

if __name__ == '__main__':
    print('Fetching the cifar dataset')
    (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
    np.save('cifar16.npy', x_train[:16])

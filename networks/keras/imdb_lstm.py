def scale_dataset(x_train):
    return x_train


def build_model():
    import os
    from keras.models import Sequential
    from keras.layers import Dense, Embedding
    from keras.layers import LSTM

    model = Sequential()
    model.add(Embedding(20000, 128, input_length=80))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    this_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(this_dir, 'networks', 'keras', 'imdb_lstm.h5')
    model.load_weights(weights_path)

    return model

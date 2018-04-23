#
#   Sentiment Analysis: Recurrent Neural Networks
#   Written by Qhan
#   2018.4.21
#

import numpy as np
import os
import os.path as osp

from keras import backend as K
from keras.layers import Embedding, Flatten
from keras.layers import LSTM, GRU, Conv1D
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def f1(Y, Yp):
    thresh = 0.0
    Yp = K.cast(K.greater(Yp, thresh), dtype='float32')
    Y = K.cast(K.greater(Y, thresh), dtype='float32')
    tp = K.sum(Y * Yp)
    
    precision = tp / (K.sum(Yp))
    recall = tp / (K.sum(Y))
    return 2 * ((precision * recall) / (precision + recall))


class RNN():
    
    def __init__(self, mp):
        self.model_path = mp if mp[-3:] == '.h5' else mp + '.h5'
        self.load_model()

    def config_model(self):
        self.n_inputs = len(self.model.inputs)
        self.mtype = self.model.layers[2].name

    def load_model(self):
        mp = self.model_path
        self.model = None
        if osp.exists(mp):
            self.model = load_model(mp, custom_objects={'f1': f1})
            self.config_model()

    def rename_model(self, new_name):
        old_path = self.model_path
        dname, fname = osp.dirname(old_path), osp.basename(old_path)
        name, ext = osp.splitext(fname)
        if new_name[-3:] == '.h5': 
            new_path = dname + '/' + new_name
        else:
            new_path = dname + '/' + new_name + ext

        os.rename(old_path, new_path)
        self.model_path = new_path

    def print_model(self):
        self.model.summary()
    
    def build_rnn(self, x, mtype, dim=128):
        if mtype == 'GRU':
            x = GRU(dim, activation='relu', dropout=0.2, name='GRU')(x)
        elif mtype == 'LSTM':
            x = LSTM(dim, activation='relu', dropout=0.2, name='LSTM')(x)
        elif mtype == 'Conv1D':
            x = Conv1D(dim, 3, activation='relu', name='Conv1D')(x)
            x = Flatten(name='Flatten')(x)
        return x

    def build_model(self, maxlen, embedding_matrix, mtype='LSTM', n_inputs=2):
        name, ext = osp.splitext(osp.basename(self.model_path))
        if ext != ".h5": self.model_path += ".h5"
        n_words, emb_dim = embedding_matrix.shape

        in_emb = Input(shape=(maxlen,), name='Input-Text')
        in_senti = Input(shape=(1,), name='Input-Sentiment')
        
        x = Embedding(n_words, emb_dim, weights=[embedding_matrix],
                      input_length=maxlen, trainable=False, name='Embedding')(in_emb)
        x = self.build_rnn(x, mtype)
        x = Dense(64, activation='relu', name='Dense')(x)
        x = Dropout(0.2, name='Dropout')(x)
        if n_inputs > 1: 
            x = Concatenate(name='Concatenate')([x, in_senti])
        out = Dense(1, activation='tanh', name='Tanh')(x)

        model = Model(inputs=[in_emb, in_senti][:n_inputs], outputs=out)
        model.compile(optimizer='adam', loss='mse', metrics=[f1])
        model.save(self.model_path)

        self.model = model
        self.config_model()

    def train(self, train_X, train_Y, val_X, val_Y, epochs=30, batch_size=32, patience=5):
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
        cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, save_weights_only=False,
                             mode='min', filepath=self.model_path)

        history = self.model.fit(train_X[:self.n_inputs], train_Y, 
                                 validation_data=(val_X[:self.n_inputs], val_Y),
                                 epochs=epochs, verbose=1, batch_size=batch_size, callbacks=[es, cp])

        hist = history.history
        print('[RNN] best val MSE: %.4f' % np.min(hist['val_loss']))

        self.model = load_model(self.model_path, custom_objects={'f1': f1})

    def predict(self, test_X):
        Yp = self.model.predict(test_X[:self.n_inputs])
        return Yp.flatten()

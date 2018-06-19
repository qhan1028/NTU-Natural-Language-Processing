#
#    Semantic Relations: Model
#    Written by Qhan
#    2018.6.18
#


import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model



class SemanticRelations():
    
    def __init__(self, categories, pos_tags, dep_tags, max_sequence_len=8):
        self.num_cases = len(categories)
        self.pos_dim = len(pos_tags)
        self.dep_dim = len(dep_tags)
        self.seq_len = max_sequence_len
        
        self.model = None
        
    
    def load_model(self, path):
        self.model = load_model(path)
    
    
    def get_word_embedding_layer(self, word_index, word_vectors, max_sequence_len=8):
        embedding_dim = 300
        word_len = len(word_index) + 1
        word_embedding_matrix = np.zeros((word_len, embedding_dim))
        for word, i in word_index.items():
            word_vector = word_vectors.get(word)
            if word_vector is not None:
                word_embedding_matrix[i] = word_vector

        return Embedding(word_len, embedding_dim, weights=[word_embedding_matrix],
                         input_length=max_sequence_len, trainable=False)

    
    def get_tag_embedding_layer(self, tag_len, max_sequence_len=8):
        tag_embedding_list = []
        for i in range(tag_len):
            vector = [0] * tag_len
            vector[i] = 1
            tag_embedding_list += [vector]

        tag_embedding_matrix = np.array(tag_embedding_list)
        return Embedding(tag_len, tag_len, weights=[tag_embedding_matrix],
                         input_length=max_sequence_len, trainable=False)

        
    def construct_model(self, word_index, word_vectors):
        in_word = Input(shape=(self.seq_len,), name='Input-Word')
        in_pos = Input(shape=(self.seq_len,), name='Input-POS')
        in_dep = Input(shape=(self.seq_len,), name='Input-Dependency')

        emb_word = self.get_word_embedding_layer(word_index, word_vectors, self.seq_len)(in_word)
        emb_pos = self.get_tag_embedding_layer(self.pos_dim, self.seq_len)(in_pos)
        emb_dep = self.get_tag_embedding_layer(self.dep_dim, self.seq_len)(in_dep)

        lstm_word = LSTM(128, activation='relu', dropout=0.2, name='LSTM-Word')(emb_word)
        lstm_pos = LSTM(128, activation='relu', dropout=0.2, name='LSTM-POS')(emb_pos)
        lstm_dep = LSTM(128, activation='relu', dropout=0.2, name='LSTM-Dependency')(emb_dep)

        x = Concatenate(name='Concatenate')([lstm_word, lstm_pos, lstm_dep])
        x = Dense(64, activation='relu', name='Dense')(x)
        x = Dropout(0.2, name='Dropout')(x)
        out = Dense(self.num_cases, activation='softmax', name='Output')(x)

        model = Model(inputs=[in_word, in_pos, in_dep], outputs=out)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.summary()
        
        self.model = model
    
    
    def train_model(self, train_data, test_data, epochs=30, batch_size=32, patience=5, save_path='../model/lstm.h5'):
        train_X = [train_data[k] for k in ['path_seq', 'pos_seq', 'dep_seq']]
        train_Y = train_data['labels']

        test_X = [test_data[k] for k in ['path_seq', 'pos_seq', 'dep_seq']]
        test_Y = test_data['labels']

        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
        cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, save_weights_only=False,
                             mode='min', filepath=save_path)

        history = self.model.fit(train_X, train_Y, validation_data=(test_X, test_Y),
                                 epochs=epochs, verbose=1, batch_size=batch_size, callbacks=[es, cp])

        return history.history
    
    
    def predict(self, data):
        X = [data[k] for k in ['path_seq', 'pos_seq', 'dep_seq']]
        one_hot = self.model.predict(X)
        seq = np.argmax(one_hot, axis=1)
        return seq
    
    
    def output_prediction(self, index, pred, case_list, save_path='../res/proposed_answer.txt'):
        with open(save_path, 'w') as f:
            for i, y in zip(index, pred):
                f.write('%d\t%s\n' % (i, case_list[y]))
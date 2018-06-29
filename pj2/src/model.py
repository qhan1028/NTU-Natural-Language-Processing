from keras.layers import Input, Embedding, LSTM, SpatialDropout1D, Dense, Dropout, GRU
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate 
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pk
import numpy as np

categories_num = 19
max_sequence_len = 8
word_dim = 300

cases = ['Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)', 'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)', 
         'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)', 'Content-Container(e1,e2)', 'Content-Container(e2,e1)', 
         'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)', 'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
         'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)', 'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
         'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)', 'Other']
         

def get_word_vectors(word_index):
    word_vectors = {}
    with open("./data/glove.6B/glove.6B.300d.txt", 'r', encoding="UTF-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_index:
                coefs = np.asarray(values[1:], dtype='float32')
                word_vectors[word] = coefs
            else:
                continue
    
    return word_vectors
        
def get_word_embedding_layer(word_index, word_vectors, max_sequence_len=8):
    embedding_dim = 300
    word_len = len(word_index) + 1
    word_embedding_matrix = np.zeros((word_len, embedding_dim))
    for word, i in word_index.items():
        word_vector = word_vectors.get(word)
        if word_vector is not None:
            word_embedding_matrix[i] = word_vector
    
    word_embedding_layer = Embedding(word_len, embedding_dim, weights=[word_embedding_matrix],
                                     input_length=max_sequence_len, trainable=False)
    
    return word_embedding_layer
    
def get_tag_embedding_layer(tag_len):
    tag_embedding_list = []
    for i in range(tag_len):
        vector = [0] * tag_len
        vector[i] = 1
        tag_embedding_list += [vector]
        
    tag_embedding_matrix = np.array(tag_embedding_list)
    tag_embedding_layer = Embedding(tag_len, tag_len, weights=[tag_embedding_matrix],
                                    input_length=max_sequence_len, trainable=False)
    
    return tag_embedding_layer

    
def construct_gru_model(word_index, word_vectors, pos_tag, dep_tag):
    in_word = Input(shape=(max_sequence_len,), name='Input-Word')
    in_pos = Input(shape=(max_sequence_len,), name='Input-POS')
    in_dep = Input(shape=(max_sequence_len,), name='Input-Dependency')
    
    emb_word = get_word_embedding_layer(word_index, word_vectors, max_sequence_len)(in_word)
    emb_pos = get_tag_embedding_layer(len(pos_tag))(in_pos)
    emb_dep = get_tag_embedding_layer(len(dep_tag))(in_dep)
    
    x = Concatenate(name='Concatenate')([emb_word, emb_pos, emb_dep])
    x = GRU(256, activation='relu', dropout=0.2, name='GRU')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(categories_num, activation='softmax', name='Output')(x)
    
    model = Model(inputs=[in_word, in_pos, in_dep], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
def construct_lstm_model(max_sequence_len, word_dim, pos_dim, dep_dim, categories,
                    word_index, word_vectors):
    
    in_word = Input(shape=(max_sequence_len,), name='Input-Word')
    in_pos = Input(shape=(max_sequence_len,), name='Input-POS')
    in_dep = Input(shape=(max_sequence_len,), name='Input-Dependency')
    
    emb_word = get_word_embedding_layer(word_index, word_vectors, max_sequence_len)(in_word)
    emb_pos = get_tag_embedding_layer(pos_dim)(in_pos)
    emb_dep = get_tag_embedding_layer(dep_dim)(in_dep)
    
    lstm_word = LSTM(512, activation='relu', dropout=0.4, name='LSTM-Word')(emb_word)
    lstm_pos = LSTM(64, activation='relu', dropout=0.2, name='LSTM-POS')(emb_pos)
    lstm_dep = LSTM(256, activation='relu', dropout=0.2, name='LSTM-Dependency')(emb_dep)
    
    x = Concatenate(name='Concatenate')([lstm_word, lstm_pos, lstm_dep])
    x = Dense(512, activation='relu', name='Dense')(x)
    x = Dropout(0.4, name='Dropout')(x)
    out = Dense(categories, activation='softmax', name='Output')(x)
    
    model = Model(inputs=[in_word, in_pos, in_dep], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model    
    
    
def construct_pooled_model(word_index, word_vectors, pos_tag, dep_tag):
    in_word = Input(shape=(max_sequence_len,), name='Input-Word')
    in_pos = Input(shape=(max_sequence_len,), name='Input-POS')
    in_dep = Input(shape=(max_sequence_len,), name='Input-Dependency')
    
    emb_word = get_word_embedding_layer(word_index, word_vectors, max_sequence_len)(in_word)
    emb_word = SpatialDropout1D(0.5)(emb_word)
    emb_pos = get_tag_embedding_layer(len(pos_tag))(in_pos)
    emb_dep = get_tag_embedding_layer(len(dep_tag))(in_dep)
    
    x = Concatenate(name='Concatenate')([emb_word, emb_pos, emb_dep])
    x = GRU(1024, return_sequences=True)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(2048, activation='selu')(conc)
    conc = Dropout(0.35)(conc)
    conc = Dense(1024, activation='selu')(conc)
    conc = Dropout(0.35)(conc)

    outp = Dense(categories_num, activation="softmax")(conc)
    model = Model(inputs=[in_word, in_pos, in_dep], outputs=outp)

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    
    return model
    
def train_model(model, model_path, train_X, train_Y, test_X, test_Y, epochs, batch_size):
    es = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max')
    cp = ModelCheckpoint(monitor='val_acc', save_best_only=True, save_weights_only=False,
                         mode='max', filepath=model_path)
    
    history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y),
                         epochs=10, verbose=1, batch_size=100, callbacks=[es,cp])

    return model

    
def make_prediction(model, result_path, test_X, test_Y):
    
    pred = np.argmax(model.predict(test_X), axis=1)
    ans = np.argmax(test_Y, axis=1)
    sum = 0
    for i in range(pred.shape[0]):
        sum += 1 if pred[i]==ans[i] else 0
    print("accuracy=", (sum+0.0)/pred.shape[0])

    with open(result_path, "w") as f:
        idx = 8001
        for i in range(pred.shape[0]):
            f.write("{0}\t{1}\n".format(idx, cases[pred[i]]))
            idx += 1

            
def double_label(labels):
    x, y = labels.shape
    new_label = np.zeros((2*x, y))
    for i in range(x):
        label = np.argmax(labels[i])
        new_label[2*i][label] = 1
        if label == len(cases)-1:
            new_label[2*i+1][len(cases)-1] = 1
        elif label % 2 == 0:
            new_label[2*i+1][label+1] = 1
        else:
            new_label[2*i+1][label-1] = 1
    return new_label
    

if __name__=="__main__":
    with open("data.pickle", "rb") as f:
        alldata = pk.load(f)
    train_data = alldata[0]
    test_data = alldata[1]
    word_tokenizer = alldata[2]
    pos_tag = alldata[3]
    dep_tag = alldata[4]

    word_index = word_tokenizer.word_index
    word_vectors = get_word_vectors(word_index)

    with open("new_data.pickle", "rb") as f:
        all_data = pk.load(f)
        train_data2 = all_data[0]
        train_X = [train_data2[k] for k in ['path_seq', 'pos_seq', 'dep_seq']]
    
    train_Y = train_data['labels']
    train_Y = double_label(train_Y)
    test_X = [test_data[k] for k in ['path_seq', 'pos_seq', 'dep_seq']]
    test_Y = test_data['labels']

    epochs = 10
    batch_size = 64
    model_path = "./model/gru.h5"
    result_path = "./result/gru_result.txt"
    model = construct_gru_model(word_index, word_vectors, pos_tag, dep_tag)
    model = train_model(model, model_path, train_X, train_Y, test_X, test_Y, epochs, batch_size)
    make_prediction(model, result_test_X, test_Y)
    #model.save("./model/gru.h5")

    epochs = 10
    batch_size = 64
    model_path = "./model/lstm.h5"
    result_path = "./result/lstm_result.txt"
    model = construct_lstm_model(max_sequence_len, word_dim, len(pos_tag), len(dep_tag), categories_num, word_index, word_vectors)
    model = train_model(model, model_path, train_X, train_Y, test_X, test_Y, epochs, batch_size)
    make_prediction(model, result_path, test_X, test_Y)
    #model.save("./model/lstm.h5")

    epochs = 30
    batch_size = 100
    model_path = "./model/pooled_lstm.h5"
    result_path = "./result/pooled_result.txt"
    model = construct_pooled_model(word_index, word_vectors, pos_tag, dep_tag)
    model = train_model(model, model_path, train_X, train_Y, test_X, test_Y, epochs, batch_size)
    make_prediction(model, result_path, test_X, test_Y)
    #model.save("./model/pooled.h5")

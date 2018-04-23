#
#   Sentiment Analysis: Data Parser
#   Written by Qhan
#   2018.4.20
#

import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#
#   Self Defined Feature
#

class GeneralSentiments():
    
    def __init__(self, corpus, word_dict):
        self.compute_sentiments(corpus, word_dict)
        
    def compute_sentiments(self, corpus, word_dict):
        print('[Preprocess] compute general sentiments', flush=True)
        
        S = []
        for data in corpus:
            S_data = 0.0
            sentences = re.split('[.,!?]', re.sub(r'#[A-Za-z]*', '', data))
            
            for sentence in sentences:
                S_sentence, v = 1., 0.
                words = [w for w in re.split(' ', sentence) if w != '']
                
                for word in words:
                    try:
                        s = word_dict.loc[word.lower()]['market_sentiment']
                        S_sentence *= s
                        v += 1
                    except:
                        pass
                    
                if v > 0:
                    S_data += np.sign(S_sentence) * np.abs(S_sentence) ** (1/v)
                    
            S.append(S_data)
        
        self.general_sentiments = np.array(S)
    
    def get_data(self):
        return self.general_sentiments

    
#   
#   Data Parser
#

class DataParser(GeneralSentiments):
    
    def __init__(self, data_dir, model_dir):
        if not osp.exists(model_dir): os.mkdir(model_dir)
        self.data_dir = data_dir if data_dir[-1] == '/' else data_dir + '/'
        self.model_dir = model_dir if model_dir[-1] == '/' else model_dir + '/'
        
        print('[Init] data dir: ' + self.data_dir + ', model dir: ' + self.model_dir)
        
        self.read_dataset()
        self.read_dictionary()
        
        self.clean_corpus()
        self.tokenize_texts()
        self.construct_embedding_matrix()
        self.compute_general_sentiments()
        
        print('[Init] done.')
        
    
    #
    #   Read training set, test set, dictionaries
    #
    
    def read_dataset(self):
        train_path = self.data_dir + 'training_set.json'
        test_path = self.data_dir + 'test_set.json'
        
        with open(train_path, 'r') as f:
            print('[Read] ' + train_path, flush=True)
            training_set = pd.DataFrame(json.load(f))
        with open(test_path, 'r') as f:
            print('[Read] ' + test_path, flush=True)
            test_set = pd.DataFrame(json.load(f))
    
        self.train_raw = training_set['tweet'].tolist()
        self.train_Y = np.array(training_set['sentiment'].astype(np.float32).tolist())
        self.train_len = len(self.train_raw)
        
        self.test_raw = test_set['tweet'].tolist()
        self.test_Y = np.array(test_set['sentiment'].astype(np.float32).tolist())
        self.test_len = len(self.test_raw)
    
    def read_dictionary(self):
        word_dict_path = self.data_dir + 'NTUSD_Fin_word_v1.0.json'
        hashtag_path = self.data_dir + 'NTUSD_Fin_hashtag_v1.0.json'
        
        with open(word_dict_path, 'r') as f:
            print('[Read] ' + word_dict_path, flush=True)
            self.word_dict = pd.DataFrame(json.load(f)).set_index('token')
        with open(hashtag_path, 'r') as f:
            print('[Read] ' + hashtag_path, flush=True)
            self.hashtag_dict = pd.DataFrame(json.load(f)).set_index('token')
    
    
    #
    #   sub-preprocess 1: clean corpus.
    #
    
    def clean_corpus(self):
        print('[Preprocess] clean corpus', flush=True)
        
        corpus = self.train_raw + self.test_raw
        
        for i, data in enumerate(corpus):
            data = re.sub(r'\$[A-Za-z0-9]*[ ,]?', '', data) # remove $ target
            data = re.sub(r'@[a-zA-Z0-9]*', '', data) # remove @ tag
            data = re.sub(r'http.*[a-zA-Z0-9]?', '', data) # remove url
            data = re.sub(r'&#39;', '\'', data) # fix '
            data = re.sub(r'[0-9.,]*[0-9]+', '', data) # remove numbers
            data = re.sub(r'(~?&[a-z]*;)', '', data) # remove Latex
            data = re.sub(r'["$%&()*+\-/:;<=>@\^_`{|}~…—\n\t•\[\]]|[.]+\.' , ' ', data) # remove characters
            data = re.sub(r'#', ' #', data) # split continuous hashtag
            data = re.sub(r' +', ' ', data) # remove space redundancy
            corpus[i] = data
        
        self.corpus = corpus
        self.train_X = corpus[:self.train_len]
        self.test_X = corpus[self.train_len:]
    
    #
    #   sub-preprocess 2: tokenize texts.
    #
    
    def tokenize_texts(self, corpus=None):
        print('[Preprocess] tokenize texts', flush=True)
        
        if corpus is None: corpus = self.corpus
        
        filters = '!"$%&()*+,-./:;<=>?@[\]^_`{|}~'
        tokenizer = Tokenizer(filters=filters)
        wi_path = self.model_dir + 'word_index.json'
        
        if not osp.exists(wi_path):
            print('[Preprocess] construct word index', flush=True)
            tokenizer.fit_on_texts(corpus)
            word_index = tokenizer.word_index
            with open(wi_path, 'w') as f:
                print('[Preprocess] save word index: ' + wi_path, flush=True)
                json.dump(word_index, f)
        else:
            with open(wi_path, 'r') as f:
                print('[Preprocess] load word index: ' + wi_path, flush=True)
                word_index = json.load(f)
            tokenizer.word_index = word_index

        train_Xi = tokenizer.texts_to_sequences(self.train_X)
        test_Xi = tokenizer.texts_to_sequences(self.test_X)
            
        self.train_Xi = pad_sequences(train_Xi)
        self.maxlen = self.train_Xi.shape[1]
        self.test_Xi = pad_sequences(test_Xi, maxlen=self.maxlen)
        
        self.word_index = word_index

    #
    #   sub-preprocess 3: construct embedding matrix.
    #
        
    def construct_embedding_matrix(self, word_index=None):
        print('[Preprocess] construct embedding matrix', flush=True)
        
        if word_index is None: word_index = self.word_index
            
        num_words = len(word_index) + 1
        emb_dim = 300 + 3
        embedding_matrix = np.zeros((num_words, emb_dim))

        for (word, index) in word_index.items():
            try:
                if word[0] == '#':
                    content = self.hashtag_dict.loc[word[1:]]    
                else:
                    content = self.word_dict.loc[word]

                bear = content['bear_cfidf'] / 100
                bull = content['bull_cfidf'] / 100
                sentiment = content['market_sentiment']
                word_vec = content['word_vec']
                embedding_matrix[index] = np.asarray(word_vec + [bear, bull, sentiment], dtype=np.float32)
            except:
                pass
        
        self.embedding_matrix = embedding_matrix
        self.num_words = num_words
        self.emb_dim = emb_dim
    
    #
    #   sub-preprocess 4: compute general sentiments.
    #
    
    def compute_general_sentiments(self):
        super(DataParser, self).__init__(self.corpus, self.word_dict)
        corpus_S = super(DataParser, self).get_data()
        self.train_S = corpus_S[:self.train_len]
        self.test_S = corpus_S[self.train_len:]


#
#   Evaluation Metrics
#

class Metrics():
    
    def mse(self, Yp, Ygt):
        return np.mean((Ygt - Yp) ** 2)
    
    def rmse(self, Yp, Ygt):
        return np.sqrt(self.mse(Ygt, Yp))

    def classify(self, Y, thres=0.):
        res = np.zeros(len(Y))
        res[Y > thres] = 1
        res[Y < thres] = -1
        res[(res != 1) & (res != -1)] = 0
        return res

    def accuracy(self, Yp, Ygt, thres=0.):
        Cgt = self.classify(Ygt, thres)
        Cp = self.classify(Yp, thres)
        return np.sum((Cgt == Cp)) / len(Yp)
    
    def f1_score(self, Yp, Ygt, thres=0.):
        Cgt = self.classify(Ygt, thres)
        Cp = self.classify(Yp, thres)
        
        true = (Cgt == Cp)
        tp, tn = true & (Cgt > 0), true & (Cgt < 0)
        
        precision = np.sum(tp) / np.sum(Cp[Cp > 0])
        recall = np.sum(tp) / np.sum(Cgt[Cgt > 0])
        
        return 2 * (precision * recall) / (precision + recall)
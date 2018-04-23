#
#   Fine-Grained Sentiment Analysis on Financial Microblogs
#   Written by Qhan
#   2018.4.20
#

import os
import os.path as osp
import numpy as np

from utils import DataParser, Metrics
from rnn import RNN


class SentimentAnalyzer(DataParser, Metrics):
    
    def __init__(self, data_dir, model_dir):
        # read data, preprocess
        super().__init__(data_dir, model_dir)
        
        self.model_dir = model_dir if model_dir[-1] == '/' else model_dir + '/'
        self.load_existed_models()
        
    def load_existed_models(self):
        mdir = self.model_dir
        models = {}
        
        for fname in os.listdir(mdir):
            if fname[-3:] == '.h5':
                print('[Load]', fname)
                rnn = RNN(mdir + fname)
                if rnn.model is not None:
                    models[fname] = rnn
                
        self.models = models
        print('[Load] done.')

    def add_model(self, fname, mtype, n_inputs):    
        rnn = RNN(self.model_dir + fname)
        rnn.build_model(self.maxlen, self.embedding_matrix, mtype, n_inputs)
        self.models[fname] = rnn

    def train_model(self, fname, epochs=30, batch_size=32, patience=5):
        train_X = [self.train_Xi, self.train_S]
        train_Y = self.train_Y
        val_X = [self.test_Xi, self.test_S]
        val_Y = self.test_Y
        self.models[fname].train(train_X, train_Y, val_X, val_Y,
                                 epochs, batch_size, patience)

    def print_model(self, fname):
        self.models[fname].print_model()

    def rename_model(self, fname, new_name):
        try:
            print('[Rename] %s -> %s' % (fname, new_name))
            self.models[fname].rename_model(new_name)
            self.models[new_name] = self.models[fname]
            self.models.pop(fname, None)
        except:
            print('[Rename] model not found.')

    def predict(self, fname, test_data=[]):
        if len(test_data) == 0:
            test_data = [self.test_Xi, self.test_S]
        
        try:
            return self.models[fname].predict(test_data)
        except:
            print('[Predict] model not found.')
            return None
        
    def ensemble(self, model_list=[], test_data=[]):
        if len(model_list) == 0:
            model_list = self.get_model_names()
            
        for i, model_name in enumerate(model_list):
            print('[Ensemble]', i, model_name)
            res = self.predict(model_name, test_data)
            if i:
                sum_res += res
            else:
                sum_res = res.copy()
        
        return sum_res / len(model_list)
    
    def evaluate(self, Yp, Ygt=None):
        if Ygt is None: Ygt = self.test_Y
        return {'mse': self.mse(Yp, Ygt),
                'acc': self.accuracy(Yp, Ygt),
                'f1': self.f1_score(Yp, Ygt)}
    
    def get_model(self, fname):
        try:
            return self.models[fname]
        except:
            print('[Get] model not found.')
            return None
        
    def get_model_names(self):
        return list(self.models.keys())
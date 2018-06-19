#
#   Semantic Relations: Utils
#   Written by Qhan
#   2018.6.17
#

import numpy as np
import os
import os.path as osp
import pandas as pd
import pickle as pk
from pyquery import PyQuery as pq
import re

from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.parse.corenlp import CoreNLPDependencyParser



class Reader():
    
    
    def __init__(self, pickle_df='../data/data_df.pickle', pickle_glove='../data/glove.pickle'):
        self.cases = ['Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
                      'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)', 
                      'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
                      'Content-Container(e1,e2)', 'Content-Container(e2,e1)', 
                      'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
                      'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
                      'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
                      'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
                      'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)', 'Other']
        self.pickle_df = pickle_df
        self.pickle_glove = pickle_glove
        
        
    def get_cases(self):
        return self.cases
    
    
    def parse_sentence(self, line):
        index, sentence_raw = line.split('\t')
        entities = [pq(sentence_raw)('e%d' % i).text() for i in [1, 2]]
        sentence = re.sub(r'</?e.>|["\n]', '', sentence_raw)
        
        return index, sentence, entities

    
    def parse_case(self, line):
        if line[-1] == '\n': line = line[:-1]
        if line in self.cases:
            return self.cases.index(line)
        else:
            return -1

    
    def _read_dataset(self, data_dir, train_file, test_file, test_ans_file):
        train_data, test_data = [], []

        with open(data_dir + train_file, 'r') as f:
            for line in f.readlines():
                if '\t' in line:
                    index, sent, ent = self.parse_sentence(line)
                else:
                    case = self.parse_case(line)
                    if case >= 0:
                        train_data += [{'index': int(index), 'sentence': sent, 'entity': ent, 'label': case}]

            train_df = pd.DataFrame(train_data).set_index('index')

        # test data are splitted into two files: TEST_FILE.txt, answer_key.txt
        with open(data_dir + test_file, 'r') as f:
            for line in f.readlines():
                if '\t' in line:
                    index, sent, ent = self.parse_sentence(line)
                    test_data += [{'index': int(index), 'sentence': sent, 'entity': ent, 'label': -1}]

            test_df = pd.DataFrame(test_data).set_index('index')

        with open(data_dir + test_ans_file, 'r') as f:
            for line in f.readlines():
                if '\t' in line:
                    index, answer = line.split('\t')
                    test_df.at[int(index), 'label'] = self.parse_case(answer)

        return train_df, test_df
   
    
    def read_dataset(self, data_dir, train_file, test_file, test_ans_file):
        if osp.exists(self.pickle_df):
            print('[Read] Found pickle file:', self.pickle_df)
            with open(self.pickle_df, 'rb') as f:
                train_df, test_df = pk.load(f)
        else:
            train_df, test_df = self._read_dataset(data_dir, train_file, test_file, test_ans_file)
            with open(self.pickle_df, 'wb') as f:
                print('[Read] Dump pickle file: ' + self.pickle_df)
                pk.dump([train_df, test_df], f)
        
        return train_df, test_df


    def _read_glove(self, data_dir, glove_file):
        word_vectors = []
        with open(data_dir + glove_file, 'r', encoding='UTF-8') as f:
            for line in f:
                word, *vector = line.split()
                word_vectors += [{'word': word, 'vector': list(map(float, vector))}]

        return pd.DataFrame(word_vectors).set_index('word')

    
    def read_glove(self, data_dir, glove_file):
        if osp.exists(self.pickle_glove):
            print('[Read] Found pickle file:', self.pickle_glove)
            with open(self.pickle_glove, 'rb') as f:
                word_vectors = pk.load(f)
        else:
            word_vectors = self._read_glove(data_dir, glove_file)
            with open(self.pickle_glove, 'wb') as f:
                print('[Read] Dump pickle file: ' + self.pickle_glove)
                pk.dump(word_vectors, f)
        
        return word_vectors
        


class DependencyParser():
    
    
    def __init__(self):
        self.cdp = CoreNLPDependencyParser()
        
        
    def find_entity_index(self, tree_list, e):
        for i, d in enumerate(tree_list):
            if d[0] == e: return i
            
        return -1

    
    def find_root_path(self, tree_list, e):
        path = []
        i = self.find_entity_index(tree_list, e)
        while True:
            ent, tag, parent, arrow = tree_list[i]
            path += [[ent, tag, arrow]]
            i = int(parent)
            if arrow is 'ROOT': break
                
        return path

    
    def _merge_path(self, p1, p2):
        rp1, rp2 = p1[::-1], p2[::-1]
        max_len = min(len(p1), len(p2))

        path = []
        for i in range(max_len):
            if rp1[i] == rp2[i]:
                m1 = len(p1) - i - 1
                m2 = i
            else:
                m1 = len(p1) - i
                m2 = i - 1
                break

        path = p1[:m1] + rp2[m2:]
        return path, m1

    
    def fix_transition_and_direction(self, path, mp):
        path[mp][2] = 'end'
        for i in range(mp, len(path)-1): # shift forward the transition tag from merge point
            path[i][2] = path[i+1][2]
            path[i+1][2] = 'end'

        for i in range(len(path)): # before merge point: 1, after merge point: 0
            path[i][2] = path[i][2].split(':')[0]
            if i < mp: path[i][2] += '-inv'

        return path

    
    def merge_path(self, p1, p2):
        path, mp = self._merge_path(p1, p2)

        if len(path) == 0 or mp < 0:
            print("Can't merge two path")

        else:
            path = self.fix_transition_and_direction(path, mp)

        return path

    
    def find_dependency_tree(self, sentence):
        tree, = self.cdp.parse(sentence.split()) # CoreNLPDependencyParser
        tree_list = [('', 'ROOT', '0', 'ROOT')] + [tuple(r.split('\t')) for r in tree.to_conll(4).split('\n')][:-1]
        
        return tree_list, tree
    
    
    def find_dependency_path(self, tree_list, entities):
        e1, e2 = entities
        p1 = self.find_root_path(tree_list, e1)
        p2 = self.find_root_path(tree_list, e2)
        
        return self.merge_path(p1, p2)

    
    def split_path_content(self, path, pos_tags, dep_tags):
        path_sent, pos_seq, dep_seq = '', [], []

        for node in path:
            word, pt, dt = node
            path_sent += word + ' '
            if pt not in pos_tags: pos_tags += [pt]
            if dt not in dep_tags: dep_tags += [dt]
            pos_seq += [pos_tags.index(pt)]
            dep_seq += [dep_tags.index(dt)]

        return path_sent, pos_seq, dep_seq, pos_tags, dep_tags



class Preprocess():
    
    
    def __init__(self, pickle_data='../data/data.pickle'):
        self.init_tags()
        self.dp = DependencyParser()
        self.pickle_data = pickle_data
        
        
    def init_tags(self):
        self.pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP',
                         'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 
                         'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', '$', '``', ',']

        dep_half_tags = ['nsubj', 'nsubjpass', 'dobj', 'iobj', 'csubj', 'csubjpass', 'ccomp', 'xcomp',
                         'nmod', 'advcl', 'advmod', 'neg',
                         'vocative', 'discourse', 'expl', 'aux', 'auxpass', 'cop', 'mark', 'punct',
                         'nummod', 'appos', 'acl', 'amod', 'det',
                         'compound', 'name', 'mwe', 'foreign', 'goeswith', 
                         'conj', 'cc',
                         'case', 'list', 'dislocated', 'parataxis', 'remnant', 'reparandum',
                         'root', 'dep']
        dep_tags = []
        for tag in dep_half_tags:
            dep_tags += [tag]
            dep_tags += [tag + '-inv']

        self.dep_tags = list(np.sort(dep_tags)) + ['end']
       
    
    def get_tags(self):
        return self.pos_tags, self.dep_tags
        
        
    def get_word_tokenizer(self, train_df, test_df):
        word_tokenizer = Tokenizer()
        all_corpus = list(train_df.loc[:]['sentence']) + list(test_df.loc[:]['sentence'])
        word_tokenizer.fit_on_texts(all_corpus)
        
        return word_tokenizer
    
    
    def _preprocess_data(self, df, word_tokenizer, max_seq_len):
        data = {}
        path_sent, pos_seq, dep_seq, labels = [], [], [], []
        pos_tags, dep_tags = self.pos_tags, self.dep_tags

        for i, row in df.iterrows():
            print('\r[Preprocess] %d' % i, end='')
            tree_list, tree = self.dp.find_dependency_tree(row['sentence'])
            path = self.dp.find_dependency_path(tree_list, row['entity'])
            psent, pos, dep, pos_tags, dep_tags = self.dp.split_path_content(path, pos_tags, dep_tags)

            path_sent += [psent]
            pos_seq += [pos]
            dep_seq += [dep]
            labels += [row['label']]
        print()
        
        # update pos_tags, dep_tags
        self.pos_tags, self.dep_tags = pos_tags, dep_tags
            
        path_seq = word_tokenizer.texts_to_sequences(path_sent)
        data['path_seq'] = pad_sequences(path_seq, maxlen=max_seq_len)
        data['pos_seq'] = pad_sequences(pos_seq, maxlen=max_seq_len)
        data['dep_seq'] = pad_sequences(dep_seq, maxlen=max_seq_len)
        data['labels'] = to_categorical(labels)

        return data

    
    def preprocess_data(self, train_df, test_df, max_seq_len=8):
        if osp.exists(self.pickle_data):
            print('[Preprocess] Found pickle file:', self.pickle_data)
            with open(self.pickle_data, 'rb') as f:
                train_data, test_data, word_tokenizer, self.pos_tags, self.dep_tags = pk.load(f)    
        else:
            word_tokenizer = self.get_word_tokenizer(train_df, test_df)
            train_data = self._preprocess_data(train_df, word_tokenizer, max_seq_len)
            test_data = self._preprocess_data(test_df, word_tokenizer, max_seq_len)
            with open(self.pickle_data, 'wb') as f:
                print('[Preprocess] Dump pickle file: ' + self.pickle_data)
                pk.dump([train_data, test_data, word_tokenizer, self.pos_tags, self.dep_tags], f)
        
        return train_data, test_data, word_tokenizer
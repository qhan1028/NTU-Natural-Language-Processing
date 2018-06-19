#
#   Semantic Relations: Main Program
#   Written by Qhan
#   2018.6.17
#

from argparse import ArgumentParser
import os.path as osp
import pickle as pk

from utils import Reader, Preprocess
from sr import SemanticRelations



#
#   raw data
#
def read_data(args):
    data_dir = args['data_dir']
    train_file = args['train_file']
    test_file = args['test_file']
    test_ans_file = args['test_ans_file']
    glove_file = args['glove_file']
    
    reader = Reader(data_dir + args['pickle_df'], data_dir + args['pickle_glove'])
    
    train_df, test_df = reader.read_dataset(data_dir, train_file, test_file, test_ans_file)
    categories = reader.get_cases()
    train_index = list(train_df.index)
    test_index = list(test_df.index)
    
    word_vectors = reader.read_glove(data_dir, glove_file)
    
    return train_df, train_index, test_df, test_index, categories, word_vectors


#
#   preprocess raw data: dependency parsing, tokenizing, one-hot vectors, categorical labels.
#
def preprocess_data(args, train_df, test_df):
    data_dir = args['data_dir']
    
    preprocessor = Preprocess(data_dir + args['pickle_data'])
    
    train_data, test_data, word_tokenizer = preprocessor.preprocess_data(train_df, test_df)
    pos_tags, dep_tags = preprocessor.get_tags()
    
    return train_data, test_data, word_tokenizer, pos_tags, dep_tags


#
#   main program
#
def main(args):
    train_df, train_index, test_df, test_index, categories, word_vectors = read_data(args)
    
    train_data, test_data, word_tokenizer, pos_tags, dep_tags = preprocess_data(args, train_df, test_df)
    word_index = word_tokenizer.word_index
    
    sr = SemanticRelations(categories, pos_tags, dep_tags, max_sequence_len=8)
    sr.construct_model(word_index, word_vectors)
    sr.train_model(train_data, test_data, epochs=50, batch_size=512, save_path='../model/lstm_2.h5')
    sr.output_prediction(test_index, sr.predict(test_data), categories, save_path='../res/proposed_answer_2.txt')
    
    

if __name__ == '__main__':
    parser = ArgumentParser('Semantic Relations')
    parser.add_argument('data_dir', default='../data/', nargs='?', help='Data directory.')
    parser.add_argument('train_file', default='TRAIN_FILE.txt', nargs='?', help='Filename of train file.')
    parser.add_argument('test_file', default='TEST_FILE.txt', nargs='?', help='Filename of test file.')
    parser.add_argument('test_ans_file', default='answer_key.txt', nargs='?', help='Filename of test answer file.')
    parser.add_argument('glove_file', default='glove.6B/glove.6B.300d.txt', nargs='?', help='Word vector file.')
    parser.add_argument('pickle_df', default='data_df.pickle', nargs='?', help='Pickle of train & test DataFrame data (not preprocessed).')
    parser.add_argument('pickle_data', default='data.pickle', nargs='?', help='Pickle of train & test preprocessed data.')
    parser.add_argument('pickle_glove', default='glove.pickle', nargs='?', help='Pickle of word_vectors.')
    args = vars(parser.parse_args())
    
    main(args)
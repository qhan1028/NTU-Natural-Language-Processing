#
#   Sentiment Analyzer: Example Usage
#   Written by Qhan
#   2018.4.22
#

from sa import SentimentAnalyzer

if __name__ == '__main__':

    sa = SentimentAnalyzer(data_dir='data', model_dir='models_self')
    models = ['GRU', 'LSTM', 'Conv1D']
    for model in models:
        print(model)
        sa.add_model(model, mtype=model, 2)
        sa.train_model(model)
        sa.evaluate(sa.predict(model))
        
    print('ensemble:', sa.evaluate(sa.ensemble()))
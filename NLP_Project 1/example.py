#
#	Sentiment Analyzer: Example Usage
#   Written by Qhan
#   2018.4.22
#

from sa import SentimentAnalyzer

if __name__ == '__main__':

	sa = SentimentAnalyzer(data_dir='data', model_dir='models')
	sa.add_model('model_1', mtype='GRU', n_inputs=2)
	sa.add_model('model_2', mtype='LSTM', n_inputs=2)
	sa.add_model('model_3', mtype='Conv1D', n_inputs=2)

	for name in sa.get_model_names():
		sa.train_model(name)

	prediction = sa.ensemble()
	performance = sa.evaluate(prediction)

	print(performance)

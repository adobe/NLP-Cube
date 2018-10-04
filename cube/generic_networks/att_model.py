import dynet as dy
from tqdm import tqdm
import numpy as np
from numpy import mean, argmax
from get_data import get_data
from numpy import ceil
import sys


"""
	Attention mechanism based sequence classification model.
	Inspired from https://nlp.stanford.edu/pubs/emnlp15_attn.pdf.
"""

class AttentionClassifier(object):
	def __init__(self, num_of_classes,
	 vocab_size,
	 embedding_size=100,
	 lstm_num_of_layers=1, 
	 state_size=200, 
	 batch_size=2):

		self.embedding_size = embedding_size
		self.lstm_num_of_layers = lstm_num_of_layers
		self.num_of_classes = num_of_classes
		self.batch_size = batch_size
		self.state_size = state_size
		self.vocab_size = vocab_size + 1

	def fit(self, X_train=None, y_train=None, X_test=None, y_test=None, epochs=None):
		X_train, y_train, X_test, y_test = self._prepare_data(X_train, y_train, X_test, y_test)
		self._initialize_model()

		print('Starting training...')

		self.trainer = dy.AdamTrainer(self.model)
		losses = []

		for _ in tqdm(range(epochs)):
			curr_loss = 0.0

			for X, y in zip(X_train, y_train):
				y_prob = self._predict_proba(X)

				loss = dy.sum_batches(dy.pickneglogsoftmax_batch(y_prob, y))
				curr_loss += loss.value()

				loss.backward()
				self.trainer.update()

			losses.append(curr_loss / len(X_train))

			print('Train Loss:', losses[-1])
			self.evaluate(X_test, y_test)
			print()

		print('Done training')

	def evaluate(self, X_test=None, y_test=None):
		acc = []
		curr_loss = 0.0

		for X, y in zip(X_test, y_test):
			proba = self._predict_proba(X, train=False)
			curr_loss += dy.sum_batches(dy.pickneglogsoftmax_batch(proba, y)).value()

			proba = proba.npvalue()

			for i in range(len(proba[0])):
				y_pred = argmax(proba[:, i])
				y_true = y[i]

				acc.append(1 if y_pred == y_true else 0)

		curr_loss /= len(X_test)

		print('Validation Loss:', curr_loss)
		print('Test Accuracy:', mean(acc))


	"""
		Defining the model's trainable parameters. We use one layer of LSTM/GRUs to process
		the sequences, on top of which we place a predictive attention mechanism to emphasize
		the words that are most relevant. The resulting context vector, which acts like a summary
		is projected, via multiplication with a learnable matrix, to the label space.
	"""
	def _initialize_model(self):
		self.model = dy.Model()
		self.input_lookup = self.model.add_lookup_parameters((self.vocab_size, self.embedding_size))

		# Attention params
		self.attention_w1 = self.model.add_parameters((self.state_size, self.state_size))
		self.attention_w2 = self.model.add_parameters((self.state_size, self.state_size))
		self.attention_v = self.model.add_parameters((1, self.state_size))

		# Predictive allignment params
		self.w_p = self.model.add_parameters((self.state_size, self.state_size))
		self.v_p = self.model.add_parameters((1, self.state_size))

		# LSTM/GRU and last layer projection matrix
		self.lstm = dy.GRUBuilder(self.lstm_num_of_layers, 
										self.embedding_size, self.state_size, self.model)

		self.output_w = self.model.add_parameters((self.num_of_classes, self.state_size))
		self.output_b = self.model.add_parameters((self.num_of_classes))

	"""
		Predictive allignments based on importance of words to each class. Here W_p and v_p are
		learnable parameters that get adjusted during training.
	"""
	def _predictive_allignment(self, sent_len, input_vector):
		w_p = dy.parameter(self.w_p)
		v_p = dy.parameter(self.v_p)

		return sent_len * dy.logistic(v_p * dy.tanh(w_p * input_vector))

	def _get_allignments(self, input_vectors):
		gaussian = lambda s, p_t, sigma : np.exp(-(s - p_t) ** 2 / (2 * sigma ** 2))

		predictive_allignments = [gaussian(len(input_vectors), self._predictive_allignment(len(input_vectors), input_vector).value()[0], len(input_vectors) / 2)
		for input_vector in input_vectors]

		monotonic_allignments = [gaussian(len(input_vectors), i, len(input_vectors)  / 2)
		for i in range(len(input_vectors))]

		allignments = {'predictive_allignments' : predictive_allignments,
		'monotonic_allignments': monotonic_allignments}

		return allignments

	"""
		 Self-attention inspired model that uses predictive/monotonic allignments
		 (best results with predictive allignments). We create an aggregate output by summing
		 the outputs of all LSTM/GRUs in the sequence, weighted by their respective attention and allignment
		 weights.
	"""
	def _attend(self, input_vectors, last_state):
		w1 = dy.parameter(self.attention_w1)
		w2 = dy.parameter(self.attention_w2)
		v = dy.parameter(self.attention_v)

		w2dt = w2 * last_state

		attention_weights = [v * dy.tanh(w1 * input_vector + w2dt) for input_vector in input_vectors]
		attention_weights = dy.softmax(dy.concatenate(attention_weights))

		allignments = self._get_allignments(input_vectors)['predictive_allignments']

		output_vector = dy.esum([vector * attention_weight * allignment
		for vector, attention_weight, allignment
		in zip(input_vectors, attention_weights, allignments)])

		return output_vector

	"""
		Prediction probabilities for a particular batch.
	"""
	def _predict_proba(self, batch, train=True, attend=True):
		dy.renew_cg()

		embedded = [dy.lookup_batch(self.input_lookup, chars) for chars in zip(*batch)]
		if train:
			self.lstm.set_dropout(0.4)
		else:
			self.lstm.disable_dropout()

		state = self.lstm.initial_state()

		output_vecs = state.transduce(embedded)
		last_state = output_vecs[-1]

		w = dy.parameter(self.output_w)
		b = dy.parameter(self.output_b)

		return w * self._attend(output_vecs, last_state) + b

	"""
		Convert the dataset to batches of given size.
	"""
	def _to_batch(self, X, y):
		data = list(zip(*sorted(zip(X, y), key=lambda x: len(x[0]))))

		batched_X = []
		batched_y = []

		for i in range(int(ceil(len(X) / self.batch_size))):
			batched_X.append(data[0][i * self.batch_size : (i + 1) * self.batch_size])
			batched_y.append(data[1][i * self.batch_size : (i + 1) * self.batch_size])

		return batched_X, batched_y

	"""
		Pad sequences for efficient batching.
	"""
	def _pad_batch(self, batch):
		max_len = len(batch[-1])
		padded_batch = []

		for x in batch:
			x = [self.vocab_size - 1] * (max_len - len(x)) + x
			padded_batch.append(x)

		return padded_batch

	"""
		Prepare data for training and testing.
	"""
	def _prepare_data(self, X_train, y_train, X_test, y_test):
		X_train, y_train = self._to_batch(X_train, y_train)
		X_test, y_test = self._to_batch(X_test, y_test)

		X_train = list(map(self._pad_batch, X_train))
		X_test = list(map(self._pad_batch, X_test))

		return X_train, y_train, X_test, y_test

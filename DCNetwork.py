from keras.models import Sequential
from keras.layers import *
from keras.callbacks import TensorBoard
from keras.optimizers import *
import numpy as np
import os

class DCNetwork:
	
	def __init__(self, actions, alpha, gamma, load_path):
		self.actions = ['down', 'right', 'up', 'left', 'nothing']
		self.model = Sequential()
		self.actions = actions
		self.gamma = gamma
		self.alpha = alpha
		self.dropout_prob = 0.1
		self.tensorboard = TensorBoard(log_dir='./output', histogram_freq=0, write_graph=False)

		self.model.add(Convolution2D(32, 8, 8, border_mode='valid', subsample=(4, 4), input_shape=(2, 84, 84)))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(64, 4, 4, border_mode='valid', subsample=(2, 2)))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())
		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.model.add(Dense(256))
		self.model.add(Activation('relu'))
		self.model.add(Dense(128))
		self.model.add(Activation('relu'))
		self.model.add(Dense(self.actions))

		self.optimizer = Adam()
		if load_path != '':
			self.load(load_path)
		if not os.path.exists('./output'):
			os.makedirs('./output')

		self.model.compile(loss = 'mean_squared_error', optimizer = self.optimizer, metrics = ['accuracy'])

	def train(self, batch):
		# Generate the xs and targets for the given batch, train the model on them 
		x_train = []
		t_train = []

		# Generate training set and targets
		for datapoint in batch:
			x_train.append(datapoint['source'])

			# Get the current Q-values for the next state and select the best
			next_state_pred = list(self.predict(datapoint['dest']).squeeze())
			next_a_idx = np.argmax(next_state_pred)
			next_a_Q_value = next_state_pred[next_a_idx]

			# Set the target so that error will be 0 on all actions except the one taken
			t = list(self.predict(datapoint['source'])[0])			
			t[datapoint['action']] = (datapoint['reward'] + self.gamma * next_a_Q_value) if not datapoint['final'] else datapoint['reward']
			
			t_train.append(t)

		print next_state_pred # Print a prediction so to have an idea of the Q-values magnitude
		x_train = np.asarray(x_train).squeeze()
		t_train = np.asarray(t_train).squeeze()
		self.model.fit(x_train, t_train, batch_size=32, nb_epoch=5, callbacks=[self.tensorboard])


	def predict(self, state):
		# Feed state into the model, return predicted Q-values
		return self.model.predict(state, batch_size=1)

	def save(self, path):
		# Save the model and its weights to disk
		print 'Saving...'
		self.model.save_weights(path)

	def load(self, path):
		# Load the model and its weights from path
		print 'Loading...'
		self.model.load_weights(path)

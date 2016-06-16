from DQNetwork import DQNetwork
import random
import numpy as np


class DQAgent:
	def __init__(self,
				 actions,
				 training_freq = 1500,
				 batch_size = 1024,
				 alpha = 0.01,
				 gamma = 0.99,
				 epsilon = 1,
				 epsilon_rate = 0.9,
				 network_input_shape = (2,84,84),
				 load_path = '',
				 logger=None):

		self.actions = actions # Size of the discreet action space
		# Training parameters
		self.training_freq = training_freq  # Number of experiences after which to train the model
		self.batch_size = batch_size
		# Hyperparameters
		self.alpha = alpha  # Learning rate (The Adam optimizer paper suggests 0.001 as default)
		self.gamma = gamma  # Discount factor
		self.epsilon = epsilon  # Coefficient for epsilon-greedy exploration
		self.epsilon_rate = epsilon_rate  # (inverse) Rate at which to make epsilon smaller, as training improves the agent's performance; epsilon = epsilon * rate
		# Experience variables
		self.experiences = []
		self.training_count = 0

		# Instantiate the deep Q-network
		self.DCN = DQNetwork(
			self.actions,
			network_input_shape,
			alpha = self.alpha,
			gamma = self.gamma,
			load_path = load_path,
			logger=logger
		)

		if logger is not None:
			logger.write({
				'Learning rate' : self.alpha,
				'Discount factor' : self.gamma,
				'Starting epsilon' : self.epsilon,
				'Epsilon decrease rate' : self.epsilon_rate,
				'Batch size' : self.batch_size
			})

	def get_action(self, state):
		# Poll DCN for Q-values, return argmax with probability 1-epsilon
		q_values = self.DCN.predict(state)
		if random.random() < self.epsilon:
			return random.randint(0,self.actions-1)
		else:
			return np.argmax(q_values)

	def add_experience(self, source, action, reward, dest, final):
		# Add a tuple (source, action, reward, dest, final) to experiences
		self.experiences.append({'source': source, 'action': action, 'reward': reward, 'dest': dest, 'final': final})

	def sample_batch(self):
		# Pop batch_size random samples from experiences and return them as a batch 
		batch = []
		for i in range(self.batch_size):
			batch.append(self.experiences.pop(random.randrange(0, len(self.experiences))))
		return np.asarray(batch)

	def must_train(self):
		# Returns true if the number of samples in experiences is greater than the batch size
		return self.batch_size <= len(self.experiences)

	def train(self):
		# Sample a batch from experiences, train the DCN on it, update the epsilon-greedy coefficient
		self.training_count += 1
		print 'Training session #', self.training_count, ' - epsilon:', self.epsilon
		batch = self.sample_batch()
		self.DCN.train(batch)  # Train the DCN
		self.epsilon *= self.epsilon_rate  # Decrease the probability of picking a random action to improve exploitation

	def quit(self, save_path):
		# Stop experiencing episodes, save the DCN, quit
		if save_path != '':
			self.DCN.save(save_path)

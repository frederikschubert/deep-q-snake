from DCNetwork import DCNetwork
import random
import numpy as np

class DQAgent:
	
	def __init__(self):
		self.actions = ['down', 'right', 'up', 'left', 'nothing']
		# Metaparameters
		self.training_freq = 200 # Number of episodes after which to train the model (related to max memory usage, too)
		self.batch_size = 128
		# Hyperparameters
		self.alpha = 0.1 # Learning rate (The Adam optimizer paper suggests 0.001 ad default)
		self.gamma = 0.9 # Discount factor
		self.epsilon = 1 # Coefficient for epsilon-greedy exploration
		self.epsilon_rate = 0.99 # (inverse) Rate at which to make epsilon smaller, as training improves the agent's performance; epsilon = epsilon * rate
		# Experience variables
		self.experiences = []
		self.old_state = [] # s
		self.next_state = [] # s'
		self.reward = None
		self.action = None
		# Q-network, deep convolutional neural network to estimate action-value function
		self.DCN = DCNetwork(self.alpha, self.gamma)

	def get_action(self, state):
		# Poll DCN for Q-values, return argmax with probability 1-epsilon
		q_values = self.DCN.predict(state)
		if random.random() < self.epsilon:
			return random.choice(self.actions)
		else: 
			return self.actions[np.argmax(q_values)]


	def add_experience(self, source, action, reward, dest, final):
		# Add a tuple (source, action, reward, dest, final) to experiences
		self.experiences.append({'source':source, 'action':action, 'reward':reward, 'dest':dest, 'final':final})

	def sample_batch(self):
		# Pop batch_size random samples from experiences and return them as a batch 
		batch =[]
		for i in range(self.batch_size):
			batch.append(self.experiences.pop(random.randrange(0, len(self.experiences))))
		return np.asarray(batch)

	def must_train(self):
		# Returns true if the number of samples in experiences is greater than the batch size
		return self.batch_size <= len(self.experiences)

	def train(self):
		# Sample a batch from experiences, train the DCN on it, update the epsilon-greedy coefficient
		print 'Traning agent...'
		batch = self.sample_batch()
		self.DCN.train(batch) # Train the DCN
		self.epsilon = self.epsilon * self.epsilon_rate # Decrease the probability of picking a random action to improve exploitation

	def quit(self):
		# Stop experiencing episodes, save the DCN, quit
		self.DCN.save()
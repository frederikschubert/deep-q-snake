class DQAgent:
	
	def __init__(self):
		# Meta parameters
		self.training_freq = 25 # Number of episodes after which to train the model (related to max memory usage, too)
		self.batch_size = 100
		# Hyperparameters
		self.alpha = 0.5 # Learning rate
		self.gamma = 0.9 # Discount factor
		self.epsilon = 1 # Coefficient for epsilon-greedy exploration
		self.epsilon_rate = 0.99 # Rate at which to make epsilon smaller, as training improves the agent's performance
		# Experience variables
		self.experiences = []
		self.old_state = [] # s
		self.next_state = [] # s'
		self.reward = None
		self.action = None
		# Q-network, deep convolutional neural network
		self.DCN = DCNetwork(alpha, gamma)

	def get_action(self, state):
		# Poll DCN for Q-values, return argmax with probability 1-epsilon

	def add_experience(self, source, action, reward, dest):
		# Add a tuple (source, action, reward, dest) to experiences

	def sample_batch(self):
		# Shuffle experiences, return batch_size random samples

	def train(self):
		print 'Traning agent...'
		batch = sample_batch()
		self.DCN.train(batch) # Train the DCN
		self.epsilon = self.epsilon * self.epsilon_rate # Decrease the probability of picking a random action to improve exploitation
import matplotlib.pyplot as p
import numpy as np

class RLplotter:
	def __init__(self):
		self.step = 0
		self.episode_nb = 0
		self.x = []
		self.eps_per_step = []
		self.lengths = []
		self.rewards = []
		self.scores = []
		self.epsilons = []

		self.episode_length = 0
		self.episodes_lengths = []
		self.episode_scores = []
		self.episode_reward = 0
		self.episodes_rewards = []


	def log(self, epsilon):
		self.x.append(self.step)
		self.step += 1
		self.eps_per_step.append(self.episode_nb)
		self.episode_nb = 0
		self.lengths.append(np.mean(self.episodes_lengths))
		self.episodes_lengths = []
		self.rewards.append(np.mean(self.episodes_rewards))
		self.episodes_rewards = []
		self.scores.append(np.mean(self.episode_scores))
		self.episode_scores = []

		self.epsilons.append(epsilon)
		pass

	def episode_step(self, reward):
		self.episode_length += 1
		self.episode_reward += reward
		pass

	def episode_end(self, score):
		self.episode_nb += 1
		self.episodes_lengths.append(self.episode_length)
		self.episode_length = 0
		self.episodes_rewards.append(self.episode_reward)
		self.episode_reward = 0
		self.episode_scores.append(score)
		pass

	def plot(self):
		p.figure()
		p.plot(self.x, self.eps_per_step, label = 'Episodes in step')
		p.ylabel('Episodes in step')
		p.xlabel('Step')
		p.draw()

		p.figure()
		p.plot(self.x, self.lengths, label = 'Average episode lenght')
		p.ylabel('Average episode lenght')
		p.xlabel('Step')
		p.draw()

		p.figure()
		p.plot(self.x, self.rewards, label = 'Average reward')
		p.ylabel('Average reward')
		p.xlabel('Step')
		p.draw()

		p.figure()
		p.plot(self.x, self.scores, label='Average score')
		p.ylabel('Average score')
		p.xlabel('step')
		p.draw()

		p.figure()
		p.plot(self.x, self.epsilons, label='Epsilon')
		p.ylabel('Epsilon')
		p.xlabel('Step')
		p.draw()

		p.show()



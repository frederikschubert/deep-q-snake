import matplotlib.pyplot as p
import numpy as np
import os
import time

class Logger:
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

		if not os.path.exists('./output'):
			os.makedirs('./output')
		self.OUT_DIR = ''.join(['./output/', time.strftime('%Y%m%d-%H%M%S/')])
		if not os.path.exists(self.OUT_DIR):
			os.makedirs(self.OUT_DIR)

	def write(self, data):
		try:
			info = open(self.OUT_DIR + 'info.txt', 'a', 0)
			for k in data:
				info.write(k + ': ' + str(data[k]) + '\n')
			info.close()
		except:
			print 'Error while opening log file.'

	def to_csv(self, filename, row):
		try:
			out_file = open(self.OUT_DIR + filename, 'a')
		except IOError:
			print 'Logger:to_csv IO error while opening file'
			return
		string = ','.join([str(val) for val in row])
		string = string + '\n' if not string.endswith('\n') else ''
		out_file.write(string)
		out_file.close()

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

	def episode_step(self, reward):
		self.episode_length += 1
		self.episode_reward += reward

	def episode_end(self, score):
		self.episode_nb += 1
		self.episodes_lengths.append(self.episode_length)
		self.episode_length = 0
		self.episodes_rewards.append(self.episode_reward)
		self.episode_reward = 0
		self.episode_scores.append(score)

	def plot(self):
		p.figure()
		step = 'Episodes in step'
		p.plot(self.x, self.eps_per_step, label =step)
		p.ylabel(step)
		p.xlabel('Step')
		p.savefig(self.OUT_DIR + step + '.png')

		p.figure()
		lenght = 'Average episode lenght'
		p.plot(self.x, self.lengths, label =lenght)
		p.ylabel(lenght)
		p.xlabel('Step')
		p.savefig(self.OUT_DIR + lenght + '.png')

		p.figure()
		reward = 'Average reward'
		p.plot(self.x, self.rewards, label =reward)
		p.ylabel(reward)
		p.xlabel('Step')
		p.savefig(self.OUT_DIR + reward + '.png')

		p.figure()
		score = 'Average score'
		p.plot(self.x, self.scores, label=score)
		p.ylabel(score)
		p.xlabel('Step')
		p.savefig(self.OUT_DIR + score + '.png')

		p.figure()
		epsilon = 'Epsilon'
		p.plot(self.x, self.epsilons, label=epsilon)
		p.ylabel(epsilon)
		p.xlabel('Step')
		p.savefig(self.OUT_DIR + epsilon + '.png')




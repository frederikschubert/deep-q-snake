import gym
from DQAgent import DQAgent
from DQNetwork import DQNetwork
from Logger import Logger
import argparse

# I/O
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--environment', type=str, help='Name of the OpenAI Gym environment to use', required=False, default='Breakout-v0')
parser.add_argument('-d', '--debug', help='Run in debug mode (no output files)', action='store_false') #Change this before release, lol
args = parser.parse_args()

logger = Logger(debug = args.debug)

# Parameters

# Entities
env = gym.make(args.environment)

# Initial logging
logger.log({
	'Environment' : args.environment,
	'Action space' : env.action_space,
	'Observation shape' : env.observation_space
})

# Main loop





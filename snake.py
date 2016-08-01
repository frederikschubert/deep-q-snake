import argparse
from pygame.locals import *
from PIL import Image, ImageOps
from DQAgent import DQAgent
from Logger import Logger
import numpy as np
import pygame
import random
import sys
import os

### CONSTANTS
MAX_EPISODES_BETWEEN_TRAININGS = 1500
# Game constants
STEP = 20
APPLE_SIZE = 20
SCREEN_SIZE = 300
START_X = SCREEN_SIZE / 2 - 5 * STEP
START_Y = SCREEN_SIZE / 2 - STEP
BACKGROUND_COLOR = (0,0,0)
SNAKE_COLOR = (255,255,255)
APPLE_COLOR = (255,255,255)
ACTIONS = 4

# Agent constants
SCREENSHOT_DIMS = (84, 84)

### FUNCTIONS
def init_snake():
	# Restores the game to the intial state. Use thsi to reset the main game loop.
	global xs, ys, dirs, score, episode_length, episode_reward, applepos, s, action, state, next_state, must_die
	xs = [START_Y, START_Y, START_Y, START_Y, START_Y]
	ys = [START_X + 5 * STEP, START_X + 4 * STEP, START_X + 3 * STEP, START_X + 2 * STEP, START_X]
	dirs = random.choice([0, 1, 3])
	score = 0
	episode_length = 0
	episode_reward = 0
	must_die = False
	applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE), random.randint(0, SCREEN_SIZE - APPLE_SIZE))

	# The direction is randomly selected
	action = random.randint(0, ACTIONS - 1)
	# Initialize the states for the first experience
	state = [screenshot(), screenshot()]
	next_state = [screenshot(), screenshot()]

	# Redraw game surface
	s.fill(BACKGROUND_COLOR)
	for ii in range(0, len(xs)):
		s.blit(img, (xs[ii], ys[ii]))
	s.blit(appleimage, applepos)
	pygame.display.update()

def collide(x1, x2, y1, y2, w1, w2, h1, h2):
	# Returns true if object at (x1, y1) collides with object at (x2, y2)
	# after moving the first of (w1, h1) and the second of (w2, h2)
	if x1 + w1 > x2 and x1 < x2 + w2 and y1 + h1 > y2 and y1 < y2 + h2:
		return True
	else:
		return False

def die():
	global logger, remaining_iters, score, episode_length, episode_reward, must_test, experience_buffer, exp_backup_counter, global_episode_counter
	global_episode_counter += 1
	# If agent is stuck, kill the process and save that precious AWS time
	if global_episode_counter > MAX_EPISODES_BETWEEN_TRAININGS:
		logger.log('Shutting process down because something seems to have gone wrong during training. Please manually check that everything is OK and in case restart the training by using the -l flag.')
		DQA.quit()
		sys.exit(0)
	# Before resetting must_test, save data about the testing episode
	if must_test:
		logger.to_csv('test_data.csv', [score, episode_length, episode_reward])
		logger.log('Test episode - Score: ' + str(score) + '; steps: ' + str(episode_length))
	must_test = False # Reset this every time (only one testing episode per training session)
	if score >= 1 and episode_length >= 10:
		print 'Adding episode to experience backup; score:', score, '; episode length:', episode_length
		logger.to_csv('train_data.csv', [score, episode_length, episode_reward])
		exp_backup_counter += len(experience_buffer)
		print 'Collected', exp_backup_counter, 'experiences of', DQA.batch_size
		for exp in experience_buffer:
			DQA.add_experience(*exp)
	# Train the network after a given number of transitions if the user requested training
	if DQA.must_train() and args.train:
		exp_backup_counter = 0
		logger.log('Episodes completed to get the desired number of experience: %d' % global_episode_counter)
		global_episode_counter = 0
		if remaining_iters == 0:
			DQA.quit()
			sys.exit(0)
		DQA.train()
		remaining_iters -= 1 if remaining_iters != -1 else 0
		must_test = True # After a training session, the next episode will be a test one
		logger.log('Test episode....')
	experience_buffer = []
	# Update graphics and restart episode
	pygame.display.update()
	init_snake()

def screenshot():
	# Take a screenshot of the screen, convert it to greyscale, resize it to 60x60, convert it to matrix form
	global s, is_headless
	data = pygame.image.tostring(s, 'RGB')  # Take screenshot
	image = Image.fromstring('RGB', (SCREEN_SIZE, SCREEN_SIZE), data) # Import it in PIL
	image = image.convert('L')  # Convert to greyscale
	image = image.resize(SCREENSHOT_DIMS)
	image = ImageOps.invert(image) if is_headless else image # Don't ever, ever, EVER ask why
	image = image.convert('1')
	matrix = np.asarray(image.getdata(), dtype=np.float64).reshape(image.size[0], image.size[1])
	return matrix

### ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='train the agent.')
parser.add_argument('-l', '--load', type=str, required=False, default='', help='load the neural network from disk.')
parser.add_argument('-i', '--iterations', type=int, required=False, default=-1, help='perform ITERATIONS training iterations before quitting.')
parser.add_argument('-v', '--novideo', action='store_true', help='suppress video output (useful to train on headless servers).')
parser.add_argument('-d', '--debug', action='store_true', help='do not print anything to file and do not create the output folder.')
parser.add_argument('--gamma', type=float, required=False, default=0.9, help='custom discount factor for the environment.')
parser.add_argument('--dropout', type=float, required=False, default=0.1, help='custom dropout rate for the Q-network.')
parser.add_argument('--reward', type=str, required=False, default='N,-1,0', help='comma separated list with rewards for apple, death and life. Pass \'N\' as apple reward to use the current snake length.')
args = parser.parse_args()

remaining_iters = args.iterations
is_headless = args.novideo
if is_headless:
	os.environ['SDL_VIDEODRIVER'] = 'dummy'
	BACKGROUND_COLOR = (255, 255, 255)  # Pygame acts weird on headless servers
if args.debug:
	print 'WARNING: debug mode is enabled, output will not be saved.'
reward_function = [float(r) if r is not 'N' else None for r in args.reward.split(',')]
APPLE_REWARD = reward_function[0]
DEATH_REWARD = reward_function[1]
LIFE_REWARD = reward_function[2]

### LOGGER
logger = Logger(debug=args.debug)
logger.log({
	'Action space': ACTIONS
})
logger.log({
	'Reward apple': APPLE_REWARD if APPLE_REWARD is not None else 'snake lenght',
	'Reward death': DEATH_REWARD,
	'Reward life': LIFE_REWARD
})  # Two different writes so the rewards will be writtes sequentially to the file
logger.to_csv('test_data.csv', ['Score,Episode length,Episode reward'])
logger.to_csv('train_data.csv', ['Score,Episode length,Episode reward'])
logger.to_csv('loss_history.csv', ['Loss'])

### AGENT
DQA = DQAgent(
	ACTIONS,
	gamma=args.gamma,
	dropout_prob=args.dropout,
	load_path = args.load,
	logger=logger
)
experience_buffer = [] # This will store the SARS tuples at each episode

### STATS
score = 0
episode_length = 0
episode_reward = 0
episode_nb = 0
exp_backup_counter = 0
global_episode_counter = 0 # Keeps track of how many episodes there were between traning iterations
must_test = False

### START
# Initialize the game variables for the first time
xs = [START_Y, START_Y, START_Y, START_Y, START_Y]
ys = [START_X + 5 * STEP, START_X + 4 * STEP, START_X + 3 * STEP, START_X + 2 * STEP, START_X]
dirs = random.choice([0, 1, 3])
must_die = False
applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE), random.randint(0, SCREEN_SIZE - APPLE_SIZE))

# Set up the GUI and the game clock
pygame.init()
s = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption('Snake')
appleimage = pygame.Surface((APPLE_SIZE, APPLE_SIZE))
appleimage.fill(APPLE_COLOR)
img = pygame.Surface((STEP, STEP))
img.fill(SNAKE_COLOR)
clock = pygame.time.Clock()

# The direction is randomly selected
action = random.randint(0,ACTIONS-1)

# Initialize the states for the first experience
state = [screenshot(), screenshot()]
next_state = [screenshot(), screenshot()]

while True: # Main game loop
	episode_length += 1
	reward = LIFE_REWARD # Reward for not dying and not eating
	next_state[0] = state[1]

	# Execute game tick and poll for system events
	clock.tick()
	for e in pygame.event.get():
		if e.type == QUIT:
			DQA.quit()
			sys.exit(0)

	# Change direction according to the action
	if action == 2 and dirs != 0: # up
		dirs = 2
	elif action == 0 and dirs != 2: # down
		dirs = 0
	elif action == 3 and dirs != 1: # left
		dirs = 3
	elif action == 1 and dirs != 3: # right
		dirs = 1

	# Check if snake hit itself
	i = len(xs) - 1
	while i >= 2:
		if collide(xs[0], xs[i], ys[0], ys[i], STEP, STEP, STEP, STEP):
			must_die = True
			reward = DEATH_REWARD  # Hit itself
		i -= 1

	# Check if snake ate apple
	if collide(xs[0], applepos[0], ys[0], applepos[1], STEP, APPLE_SIZE, STEP, APPLE_SIZE):
		score += 1
		reward = APPLE_REWARD if APPLE_REWARD is not None else len(xs)
		xs.append(700)
		ys.append(700)
		applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE), random.randint(0, SCREEN_SIZE - APPLE_SIZE))  # Ate apple

	# Check if snake collided with walls
	if xs[0] < 0 or xs[0] > SCREEN_SIZE - APPLE_SIZE * 2 or ys[0] < 0 or ys[0] > SCREEN_SIZE - APPLE_SIZE * 2:
		must_die = True
		reward = DEATH_REWARD  # Hit wall

	# Move snake of 1 step in the current direction
	i = len(xs) - 1
	while i >= 1:
		xs[i] = xs[i - 1]
		ys[i] = ys[i - 1]
		i -= 1
	if dirs == 0:
		ys[0] += STEP
	elif dirs == 1:
		xs[0] += STEP
	elif dirs == 2:
		ys[0] -= STEP
	elif dirs == 3:
		xs[0] -= STEP

	# Redraw game surface
	s.fill(BACKGROUND_COLOR)
	for i in range(0, len(xs)):
		s.blit(img, (xs[i], ys[i]))
	s.blit(appleimage, applepos)
	pygame.display.update()

	# Update next state
	next_state[1] = screenshot()
	# Add SARS tuple to experience_buffer
	experience_buffer.append((np.asarray([state]), action, reward, np.asarray([next_state]), True if must_die else False))
	episode_reward += reward
	# Change current state
	state = list(next_state)
	# Poll the DQAgent to get the next action
	action = DQA.get_action(np.asarray([state]), testing=must_test)

	if must_die:
		die()

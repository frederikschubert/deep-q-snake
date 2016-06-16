from pygame.locals import *
from PIL import Image
from DQAgent import DQAgent
from logger import Logger
import numpy as np
import pygame
import random
import sys
import getopt

# Game constants
STEP = 20
APPLE_SIZE = 20
SCREEN_SIZE = 300
START_X = SCREEN_SIZE / 2 - 5 * STEP
START_Y = SCREEN_SIZE / 2 - STEP
BACKGROUND_COLOR = (0, 0, 0)
SNAKE_COLOR = (255, 0, 0)
APPLE_COLOR = (255, 255, 255)
ACTIONS = 4

# Agent constants
SCREENSHOT_DIMS = (84, 84)
APPLE_REWARD = None
DEATH_REWARD = -1
LIFE_REWARD = 0

# Argument defined variables
must_train = False
save_path = ''
load_path = ''
remaining_iters = -1

# Stats
logger = Logger()
logger.write({
	'Action space' : ACTIONS
})
logger.write({
	'Reward apple' : APPLE_REWARD if APPLE_REWARD is not None else 'snake lenght',
	'Reward death' : DEATH_REWARD,
	'Reward life' : LIFE_REWARD
}) # Two different writes so the rewards will be writtes sequentially to the file

def init_snake():
	# Restores the game to the intial state. To be used in the main game loop.
	global xs, ys, dirs, score, applepos, s, t, action, state, next_state, must_die
	xs = [START_Y, START_Y, START_Y, START_Y, START_Y]
	ys = [START_X + 5 * STEP, START_X + 4 * STEP, START_X + 3 * STEP, START_X + 2 * STEP, START_X]
	dirs = random.choice([0, 1, 3])
	score = 0
	must_die = False
	applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE), random.randint(0, SCREEN_SIZE - APPLE_SIZE))

	# First action is set to nothing (the direction is randomly selected anyway)
	action = 4 # nothing
	# Initialize the states for the first experience
	state = [screenshot(), screenshot()]
	next_state = [screenshot(), screenshot()]

	# Redraw game surface
	s.fill(BACKGROUND_COLOR)
	for ii in range(0, len(xs)):
		s.blit(img, (xs[ii], ys[ii]))
	s.blit(appleimage, applepos)
	t = f.render(str(score), True, (0, 0, 0))
	s.blit(t, (10, 10))
	pygame.display.update()


def collide(x1, x2, y1, y2, w1, w2, h1, h2):
	# Returns true if object at (x1, y1) collides with object at (x2, y2)
	# after moving the first of (w1, h1) and the second of (w2, h2)
	if x1 + w1 > x2 and x1 < x2 + w2 and y1 + h1 > y2 and y1 < y2 + h2:
		return True
	else:
		return False


def die():
	global logger, score
	logger.episode_end(score)
	pygame.display.update()
	init_snake()


def screenshot():
	# Take a screenshot of the screen, convert it to greyscale, resize it to 60x60, convert it to matrix form
	global s
	data = pygame.image.tostring(s, 'RGB')  # Take screenshot
	image = Image.fromstring('RGB', (SCREEN_SIZE, SCREEN_SIZE), data)  # Import it in PIL
	image = image.convert('L')  # Convert to greyscale
	image = image.resize(SCREENSHOT_DIMS)
	matrix = np.asarray(image.getdata(), dtype=np.float64).reshape(image.size[0], image.size[1])
	return matrix


try:
	opts, args = getopt.getopt(sys.argv[1:], 'hts:l:i:', ['help', 'train', 'save=', 'load=', 'iterations='])
except getopt.GetoptError:
	print 'Usage: snake.py [-t] [-s path/to/file.h5] [-l path/to/file.h5] [-i num_iter]'
	sys.exit(2)
for opt, arg in opts:
	if opt in ('-h', '--help'):
		print 'Usage: snake.py [-t] [-s path/to/file.h5] [-l path/to/file.h5] [-i num_iter]'
	elif opt in ('-t', '--train'):
		print 'Training...'
		must_train = True
	elif opt in ('-s', '--save'):
		save_path = arg
	elif opt in ('-l', '--load'):
		load_path = arg
	elif opt in ('-i', '--iterations'):
		remaining_iters = int(arg)

# Instantiate the agent
DQA = DQAgent(
	ACTIONS,
	load_path = load_path,
	logger=logger
)

# Initialize the game variables for the first time
xs = [START_Y, START_Y, START_Y, START_Y, START_Y]
ys = [START_X + 5 * STEP, START_X + 4 * STEP, START_X + 3 * STEP, START_X + 2 * STEP, START_X]
dirs = random.choice([0, 1, 3])
score = 0
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
f = pygame.font.SysFont('Arial', STEP)
clock = pygame.time.Clock()


# First action is set to nothing (the direction is randomly selected anyway)
action = 4 # nothing
# Initialize the states for the first experience
state = [screenshot(), screenshot()]
next_state = [screenshot(), screenshot()]

while True:
	reward = LIFE_REWARD # Reward for not dying and not eating
	next_state[0] = state[1]

	# Execute game tick and poll for system events
	clock.tick()
	for e in pygame.event.get():
		if e.type == QUIT:
			DQA.quit(save_path)
			logger.plot()
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
	t = f.render(str(score), True, (0, 0, 0))
	s.blit(t, (10, 10))
	pygame.display.update()

	# Update next state
	next_state[1] = screenshot()
	# Add <old_state, a, r, new_state, final> to experiences 
	DQA.add_experience(np.asarray([state]), action, reward, np.asarray([next_state]), True if must_die else False)
	logger.episode_step(reward)
	# Change current state
	state = list(next_state)
	# Poll the DQAgent to get the next action
	action = DQA.get_action(np.asarray([state]))
	# Train the network after a given number of transitions if the user requested training
	if DQA.must_train() and must_train:
		logger.log(DQA.epsilon)
		if remaining_iters == 0:
			DQA.quit(save_path)
			logger.plot()
			sys.exit(0)
		DQA.train()
		remaining_iters -= 1 if remaining_iters != -1 else 0

	if must_die:
		die() # Lol

import pygame, random, sys
from pygame.locals import *
from pygame.image import *
from PIL import Image
import numpy as np
from DQAgent import DQAgent

# Game constants
STEP = 10
APPLE_SIZE = 10
SCREEN_SIZE = 200
START_X = SCREEN_SIZE / 2 - 5 * STEP
START_Y = SCREEN_SIZE / 2 - STEP
FPS = 120
BACKGROUND_COLOR = (255,255,255)
SNAKE_COLOR = (255,0,0)
APPLE_COLOR = (0,0,0)

# Agent constants
SCREENSHOT_DIMS = (60, 60)
APPLE_REWARD = 100
DEATH_REWARD = -1000
LIFE_REWARD = 1

def init_snake():
	# Restores the game to the intial state. To be used in the main game loop. 
	
	global xs, ys, dirs, score, applepos, s, t, action, state, next_state, must_die
	xs = [START_Y, START_Y, START_Y, START_Y, START_Y];
	ys = [START_X + 5*STEP, START_X + 4*STEP, START_X + 3*STEP, START_X + 2*STEP, START_X];
	dirs = random.choice([0,1,3]);
	score = 0;
	must_die = False
	applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE), random.randint(0, SCREEN_SIZE - APPLE_SIZE));
	
	# First action is set to nothing (the direction is randomly selected anyway)
	action = 'nothing'
	# Initialize the states for the first experience
	state = [screenshot(), screenshot()]
	next_state = [screenshot(), screenshot()]
	
	# Redraw game surface
	s.fill(BACKGROUND_COLOR)	
	for i in range(0, len(xs)):
		s.blit(img, (xs[i], ys[i]))
	s.blit(appleimage, applepos);
	t=f.render(str(score), True, (0, 0, 0));
	s.blit(t, (10, 10));
	pygame.display.update()	

def collide(x1, x2, y1, y2, w1, w2, h1, h2):
	# Returns true if object at (x1, y1) collides with object at (x2, y2) after moving the first of (w1, h1) and the second of (w2, h2)

	if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:
		return True
	else:
		return False

def die():
	pygame.display.update();
	init_snake()

def screenshot():
	# Take a screenshot of the screen, convert it to greyscale, resize it to 60x60, convert it to matrix form

	global s
	data = pygame.image.tostring(s, 'RGB') # Take screenshot
	img = Image.fromstring('RGB', (SCREEN_SIZE, SCREEN_SIZE), data) # Import it in PIL
	img = img.convert('L') # Convert to greyscale
	img = img.resize(SCREENSHOT_DIMS)
	matrix = np.asarray(img.getdata(), dtype=np.float64).reshape(img.size[0], img.size[1])
	return matrix


# Initialize the game variables for the first time
xs = [START_Y, START_Y, START_Y, START_Y, START_Y];
ys = [START_X + 5*STEP, START_X + 4*STEP, START_X + 3*STEP, START_X + 2*STEP, START_X];
dirs = random.choice([0,1,3]);
score = 0;
must_die = False
applepos = (random.randint(0, SCREEN_SIZE - APPLE_SIZE), random.randint(0, SCREEN_SIZE - APPLE_SIZE));

# Set up the GUI and the game clock
pygame.init();
s=pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE));
pygame.display.set_caption('Snake');
appleimage = pygame.Surface((APPLE_SIZE, APPLE_SIZE));
appleimage.fill(APPLE_COLOR);
img = pygame.Surface((STEP, STEP));
img.fill(SNAKE_COLOR);
f = pygame.font.SysFont('Arial', STEP);
clock = pygame.time.Clock()

# Instantiate the agent
DQA = DQAgent()

# First action is set to nothing (the direction is randomly selected anyway)
action = 'nothing'
# Initialize the states for the first experience
state = [screenshot(), screenshot()]
next_state = [screenshot(), screenshot()]

while True:
	reward = LIFE_REWARD # Reward for not dying and not eating
	next_state[0] = state[1]

	# Execute game tick and poll for system events
	clock.tick(FPS)
	for e in pygame.event.get():
		if e.type == QUIT:
			sys.exit(0)

	# Change direction according to the action
	if action == 'up' and dirs != 0: dirs = 2
	elif action == 'down' and dirs != 2: dirs = 0
	elif action == 'left' and dirs != 1: dirs = 3
	elif action == 'right' and dirs != 3: dirs = 1

	# Check if snake hit itself
	i = len(xs)-1
	while i >= 2:
		if collide(xs[0], xs[i], ys[0], ys[i], STEP, STEP, STEP, STEP):
			must_die = True
			reward = DEATH_REWARD # Snake hit itself
		i-= 1

	#  Check if snake ate apple
	if collide(xs[0], applepos[0], ys[0], applepos[1], STEP, APPLE_SIZE, STEP, APPLE_SIZE):
		score+=1;
		reward = APPLE_REWARD
		xs.append(700);
		ys.append(700);
		applepos=(random.randint(0,SCREEN_SIZE - APPLE_SIZE),random.randint(0,SCREEN_SIZE - APPLE_SIZE)) # Snake ate APPLE_SIZE

	# Check if snake collided with walls
	if xs[0] < 0 or xs[0] > SCREEN_SIZE - APPLE_SIZE * 2 or ys[0] < 0 or ys[0] > SCREEN_SIZE - APPLE_SIZE * 2: 
		must_die = True
		reward = DEATH_REWARD # Snake hit wall

	# Move snake of 1 step in the current direction
	i = len(xs)-1
	while i >= 1:
		xs[i] = xs[i-1];ys[i] = ys[i-1];
		i -= 1

	if dirs==0:ys[0] += STEP
	elif dirs==1:xs[0] += STEP
	elif dirs==2:ys[0] -= STEP
	elif dirs==3:xs[0] -= STEP	

	# Redraw game surface
	s.fill(BACKGROUND_COLOR)	
	for i in range(0, len(xs)):
		s.blit(img, (xs[i], ys[i]))
	s.blit(appleimage, applepos);
	t=f.render(str(score), True, (0, 0, 0));
	s.blit(t, (10, 10));
	pygame.display.update()
	
	# Update next state
	next_state[1] = screenshot()
	# Add <old_state, a, r, new_state, final> to experiences 
	DQA.add_experience(np.asarray([state]), str(action), int(reward), np.asarray([next_state]), True if must_die else False)
	# Change current state
	state = list(next_state)
	# Poll the DQAgent to get the next action
	action = DQA.get_action(np.asarray([state]))
	# Train the network after a given number of transitions
	if DQA.must_train():
		DQA.train()
	if must_die :
		die() # Lol
					
					
			



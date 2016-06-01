import pygame, random, sys
from pygame.locals import *

STEP = 10
APPLE = 10
START_X = 210
START_Y = 290
SCREEN_SIZE = 600
FPS = 10

def init_snake():
	global xs, ys, dirs, score, applepos, s, t
	xs = [START_Y, START_Y, START_Y, START_Y, START_Y];
	ys = [START_X + 5*STEP, START_X + 4*STEP, START_X + 3*STEP, START_X + 2*STEP, START_X];
	dirs = random.choice([0,1,3]);
	score = 0;
	applepos = (random.randint(0, 590), random.randint(0, 590));
	s.fill((255, 255, 255))	
	for i in range(0, len(xs)):
		s.blit(img, (xs[i], ys[i]))

	s.blit(appleimage, applepos);
	t=f.render(str(score), True, (0, 0, 0));
	s.blit(t, (10, 10));
	pygame.display.update()	

def collide(x1, x2, y1, y2, w1, w2, h1, h2):
	if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:
		return True
	else:
		return False

def die(screen, score):
	f = pygame.font.SysFont('Arial', 30);
	t = f.render('Your score was: '+str(score), True, (0, 0, 0));
	screen.blit(t, (10, 270));
	pygame.display.update();
	pygame.time.wait(2000);
	init_snake()

xs = [START_Y, START_Y, START_Y, START_Y, START_Y];
ys = [START_X + 5*STEP, START_X + 4*STEP, START_X + 3*STEP, START_X + 2*STEP, START_X];
dirs = random.choice([0,1,3]);
score = 0;
applepos = (random.randint(0, 590), random.randint(0, 590));
pygame.init();
s=pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE));
pygame.display.set_caption('Snake');
appleimage = pygame.Surface((APPLE, APPLE));
appleimage.fill((0, 0, 0));
img = pygame.Surface((STEP, STEP));
img.fill((255, 0, 0));
f = pygame.font.SysFont('Arial', STEP);
clock = pygame.time.Clock()

# a = 'do_nothing'
# old_state = [screenshot, screenshot]
while True:

	# LOCAL 	next_state[0] = old_state[1]
	# POLL DQA 	Act (e = a)
	# LOCAL 	Move
	# LOCAL 	next_state[1] = screenshot
	# POLL DQA 	Get reward 
	# POLL DQA 	Add to experiences <old_state, a, r, new_state, final>
	# POLL NET  Get Q-Values, choose the argmax with a probability 1-e, otherwise any other random action
	# LOCAL 	old_state = next_state
	# POLL NET 	if current_iter % train_freq == 0 then train network on minibatch and make epsilon smaller

	clock.tick(FPS)
	for e in pygame.event.get():
		if e.type == QUIT:
			sys.exit(0)
		elif e.type == KEYDOWN:
			if e.key == K_UP and dirs != 0: dirs = 2
			elif e.key == K_DOWN and dirs != 2: dirs = 0
			elif e.key == K_LEFT and dirs != 1: dirs = 3
			elif e.key == K_RIGHT and dirs != 3: dirs = 1

	i = len(xs)-1
	while i >= 2:
		if collide(xs[0], xs[i], ys[0], ys[i], STEP, STEP, STEP, STEP):
			die(s, score)
		i-= 1

	if collide(xs[0], applepos[0], ys[0], applepos[1], STEP, APPLE, STEP, APPLE):
		score+=1;
		xs.append(700);
		ys.append(700);
		applepos=(random.randint(0,590),random.randint(0,590))

	if xs[0] < 0 or xs[0] > 580 or ys[0] < 0 or ys[0] > 580: 
		die(s, score)

	i = len(xs)-1
	while i >= 1:
		xs[i] = xs[i-1];ys[i] = ys[i-1];
		i -= 1

	if dirs==0:ys[0] += STEP
	elif dirs==1:xs[0] += STEP
	elif dirs==2:ys[0] -= STEP
	elif dirs==3:xs[0] -= STEP	

	s.fill((255, 255, 255))	
	for i in range(0, len(xs)):
		s.blit(img, (xs[i], ys[i]))

	s.blit(appleimage, applepos);
	t=f.render(str(score), True, (0, 0, 0));
	s.blit(t, (10, 10));
	pygame.display.update()
					
					
			



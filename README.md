# Playing Snake with Deep Q-Learning by Daniele Grattarola
This is an implementation in Keras and Pygame of deep Q-learning applied to Snake. 
Rather than a pre-packaged tool to simply see the agent playing the game, this is a model that needs to be trained and fine tuned by hand and has more of an educational value.   
The code used to play the game has been adapted from [this 35 LOC example](http://pygame.org/project-Snake+in+35+lines-818-.html).   
To run the script you'll need a working Keras and Pygame installation: see [here](http://keras.io/#installation), [here](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#download-and-setup) and [here](http://www.pygame.org/wiki/GettingStarted) for detailed instructions on how to install the libraries; you might want to enable GPU support in order to speed up the convolution, but since this a rather simple model it is not strictly necessary.    
### Installation
Simply download the source code:
```sh
git clone https://gitlab.com/danielegrattarola/deep-q-snake.git
cd deep-q-snake
```
and run: 
```sh
python snake.py -h
```
to see the arguments list.

### Usage
### Model description
The model is composed of a deep convolutional neural network and a classical reinforcement learning agent. The agent uses the network to approximate the environment's action-value function and decides which action to take at each timestep.
In reinforcement learning, the agent interacts with the environment by choosing an action; every action has two consequences:
- 1. the state of the environment changes
- 2. the agent is assigned a reward (positive or negative)
In this example, the environment is the game and the agent is the player. 
At any time 

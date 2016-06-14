# Playing Snake with Deep Q-Learning by Daniele Grattarola
This is an implementation in Keras and Pygame of deep Q-learning applied to Snake. 
Rather than a pre-packaged tool to simply see the agent playing the game, this is a model that needs to be trained and fine tuned by hand and has more of an educational value.   
The code used to play the game has been adapted from [this 35 LOC example](http://pygame.org/project-Snake+in+35+lines-818-.html).   

### Installation
To run the script you'll need Keras and Pygame installed on your system: see [here](http://keras.io/#installation)and [here](http://www.pygame.org/wiki/GettingStarted) for detailed instructions on how to install the libraries; you might want to enable GPU support in order to speed up the convolution, but since this a rather simple model it is not strictly necessary.    
Other dependencies are Matplotliband PIL.    

Simply download the source code:
```sh
git clone https://gitlab.com/danielegrattarola/deep-q-snake.git
cd deep-q-snake
```
and run: 
```sh
python snake.py -t
```
to run a training session on a new model.    

### Usage
By running:
```sh
python snake.py -h
```
you'll see the options list. Possible options are:
- **t, train**: if set, the model will train the deep Q-network periodically during the game. 
- **s, save path/to/file.h5**: upon exiting, the model will be saved to a HDF5 file.
- **l, load path/to/file.h5**: initialize the deep Q-network using the weights stored in the given HDF5 file.

### Model description
The model is composed of a deep convolutional neural network and a classical reinforcement learning agent. The agent uses the network to approximate the environment's action-value function and decides which action to take at each timestep.   
The deep Q-network is structured as follows:   
- **Convolution layer**: uses 8x8 filters, 32 channels, 4x4 stride and ReLU activations to perform convolution on the 2x84x84 input. Outputs a 32x20x20 volume.
- **Convolution layer**: uses 4x4 filters, 64 channels, 2x2 stride and ReLU activations to perform convolution on the 32x20x20 input. Outputs a normalized 64x9x9 volume.
- **Convolution layer**: uses 4x4 filters, 64 channels, 2x2 stride and ReLU activations to perform convolution on the 64x9x9 input. Outputs a normalized 64x7x7 volume.
- **Dense**: ReLU activation layer which outputs a 1x512 tensor.
- **Dense**: ReLU activation layer which outputs a 1x256 tensor.
- **Dense**: ReLU activation layer which outputs a 1x128 tensor.
- **Dense**: linear activation layer which outputs a 1x5 tensor.    
   
No pooling layers were used because they are mostly useful to provide translation invariance during the convolution phase. Since in this example the position of the snake is an important piece of information, it's better to only downscale the feature maps during normal convolution.   
The linear activation used in the last layer of the net is necessary because the network will try to reach both negative numbers (so relu is not a good choice) and numbers which are big in absolute value (so no tanh because it will be rounded to 1.0 annd -1.0 most of the times). I'm still working on a better solution, but for now it should work.    
More details on the network architecture and training procedure can be found in the paper.    

### Output
Running the script with any combination of options will output some useful data collections to help you interpret the data.     
When quitting the script by clicking on the close button of the snake window five plots will be displayed as a function of training steps: 
1. **Episodes in step**: how many episodes were completed before the corresponding training step
2. **Average episode length**: how long were the episodes, on average, before the corresponding training step
3. **Average reward**: how much was the agent rewarded, on average, in the episodes preceding the corresponding training step
4. **Average score**: how many apples did the snake eat, on average, in the episodes preceding the corresponding training step
5. **Epsilon**: the value of the epsilon greedy coefficient in the episodes preceding the corresponding training step   
Moreover, at every training step the Tensorboard callback will be executed by Keras and the logfile will be saved in the `'./output'` folder.
To visualize the latest graphs, simply run:

```sh
tensorboard --logdir='./output'
```
and navigate to `localhost:6006`.

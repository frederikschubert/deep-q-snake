# Playing Snake with Deep Q-Learning by Daniele Grattarola
This is an implementation in Keras and Pygame of deep Q-learning applied to Snake. 
Rather than a pre-packaged tool to simply see the agent playing the game, this is a model that needs to be trained and fine tuned by hand and has more of an educational value.   
The code used to play the game has been adapted from [this 35 LOC example](http://pygame.org/project-Snake+in+35+lines-818-.html).   
 
### Installation
To run the script you'll need Keras, h5py and Pygame installed on your system: see [here](http://keras.io/#installation), [here](http://docs.h5py.org/en/latest/build.html) and [here](http://www.pygame.org/wiki/GettingStarted) for detailed instructions on how to install the libraries; you might want to enable GPU support in order to speed up the convolution, but since this a rather simple model it is not strictly necessary.   
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
- **Convolution layer**: uses 8x8 filters on 32 channels with 4x4 stride to perform convolution on the 2x60x60 input. Outputs a normalized 32x14x14 volume.
- **Convolution layer**: uses 4x4 filters on 64 channels with 2x2 stride to perform convolution on the 32x14x14 input. Outputs a normalized 64x6x6 volume.
- **Dense**: linear activation layer which outputs a 1x512 tensor.
- **Dense**: linear activation layer which outputs a 1x256 tensor.
- **Dense**: linear activation layer which outputs a 1x128 tensor.
- **Dense**: linear activation layer which outputs a 1x5 tensor.
   
No pooling layers were used because they are mostly useful to provide translation invariance during the convolution phase. Since in this example the position of the snake is an important piece of information, it's better to only downscale the feature maps during normal convolution.   
The linear activations used in the fully connected portion of the net are necessary because the network will try to reach both negative numbers (so relu is not a good choice) and numbers which are big in absolute value (so no tanh because it will be rounded to 1.0 annd -1.0 most of the times). I'm still working on a better solution, but for now it should work.    


# Playing Snake with Deep Q-Learning by Daniele Grattarola
This is an implementation in Keras and Pygame of deep Q-learning applied to Snake. 
Rather than a pre-packaged tool to simply see the agent playing the game, this is a model that needs to be trained and fine tuned by hand and has more of an educational value.   
The code used to play the game has been adapted from [this 35 LOC example](http://pygame.org/project-Snake+in+35+lines-818-.html).   

### Installation
To run the script you'll need Keras and Pygame installed on your system: see [here](http://keras.io/#installation) and [here](http://www.pygame.org/wiki/GettingStarted) for detailed instructions on how to install the libraries; you might want to enable GPU support in order to speed up the convolution, but since this a rather simple model it is not strictly necessary.    
Other dependencies include PIL and h5py (see [here](http://packages.ubuntu.com/trusty/python-h5py) for installation on Ubuntu), which should be available through pip.   

To run the script simply download the source code:
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
- **t, train**: train the Q-network periodically during the game. 
- **l, load path/to/file.h5**: initialize the Q-network using the weights stored in the given HDF5 file.
- **i, iterations number**: perform number training iterations before quitting.
- **v, no-video**: suppress video output (useful to train on headless servers).
- **d, debug**: do not print anything to file and do not create the output folder.  
- **gamma**: set a custom value for the discount factor
- **dropout**: set a custom value for the dropout probability of the Q-network
- **reward**: set a custom reward function by passing a comma separated list with rewards for apple, death and life (pass 'N' as life reward to use the current snake lenght)

### Output
Running the script with any combination of options will output some useful data collections to help you interpret the data.     
You'll find data.csv and test_data.csv in the output folder of the run (output/runYYYMMDD-hhmmss) which will contain the score and episode length for each episode (training and testing episodes respectively).

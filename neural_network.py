## Python module to read the dataset and create the model

import numpy as np
import pandas as pd
import os
import gym
import cv2
import argparse
import sys, glob
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Convolution2D

# Load dataset
# Now we will read the dataset
df=pd.read_csv('X.txt', sep = '.', header = None)
print(df.shape)

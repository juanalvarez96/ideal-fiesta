# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# and uses Keras.
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
from keras.models import model_from_json # Needed to load out neural network
import pdb
import imageio

#Initialize
env = gym.make("Pong-v0")
number_of_inputs = env.action_space.n #This is incorrect for Pong (?)
#number_of_inputs = 1
observation = env.reset()

def pong_preprocess_screen(I):
  obs_t = I[34:194:4, 12:148:2, 1]
  return obs_t


json_file = open('model1.json', 'r')
loaded = json_file.read()
json_file.close()
model = model_from_json(loaded)



#Begin training
while True:
  #if render: 
  env.render()
  #Preprocess, consider the frame difference as features
  #pdb.set_trace()
  cur_x = pong_preprocess_screen(observation)
  action = model.predict_classes(cur_x)
  value = int(sum(action)/len(action))
  dic = {
    0:0,
    -1:3,
    1:2
  }
  value_final = dic.get(value)
  print(value_final)
  #pdb.set_trace()
  #print(value)
  observation, _,_,_ = env.step(value_final)

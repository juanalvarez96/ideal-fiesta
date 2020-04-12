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

# Get X and y vectors
X = df.iloc[:, 0:68]
y = df[68].map({2:1, 3:-1, 0:0}) # Careful: This is a pandas series


# Define keras models
def model1():
    model = Sequential()
    model.add(Dense(110, input_dim = 68, activation="relu"))
    model.add(Dense(200, input_dim = 110, activation="relu"))
    model.add(Dense(30, input_dim = 200, activation="relu"))
    model.add(Dense(8, input_dim = 30, activation="relu"))
    model.add(Dense(1, input_dim = 8, activation="sigmoid"))
    return model

def model2():
    model = Sequential()
    model.add(Dense(90, input_dim = 68, activation="relu"))
    model.add(Dense(150, input_dim = 90, activation="relu"))
    model.add(Dense(200, input_dim = 150, activation="relu"))
    model.add(Dense(150, input_dim = 200, activation="relu"))
    model.add(Dense(90, input_dim = 150, activation="relu"))
    model.add(Dense(30, input_dim = 90, activation="relu"))
    model.add(Dense(8, input_dim = 30, activation="relu"))
    model.add(Dense(1, input_dim = 8, activation="sigmoid"))
    return model


def model3():
    model = Sequential()
    model.add(Dense(159, input_dim = 68, activation="relu"))
    model.add(Dense(100, input_dim = 159, activation="relu"))
    model.add(Dense(1, input_dim = 100, activation="sigmoid"))    
    return model

model1 = model1()
model2 = model2()
model3 = model3()

models = [model1,model2, model3]
i = 0

for model in models:
    # Compile keras models
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics = ['accuracy'])

    # Fit keras model
    model.fit(X,y,epochs=7, batch_size=12)

    # Evaluate model
    _, accuracy=model.evaluate(X,y)
    print('Accuracy: %.2f' % (accuracy*100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model{}.json".format(i), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    i = i+1

    

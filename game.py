import gym
from gym.utils.play import play
import imageio
import pdb
import numpy as np
import pandas as pd


def mycallbacks(obs_t, obs_tp1, action, rew, done, info):
    # We have to add the action 
    pdb.set_trace()
    obs_t = obs_t[34:194:4, 12:148:2, 1]
    b = np.zeros((obs_t.shape[0], obs_t.shape[1]+1))
    b[:, :-1] = obs_t
    b[:,-1]= action
    #imageio.imwrite("jeu.jpg",obs_t)
    #print("Initial dim {} | Result {}".format(obs_t.shape, b.shape))
    # Check result
    print(b.shape)
    print(b)
    #with open('X.txt', 'a') as outfileX:
        #np.savetxt(outfileX,delimiter='.', X=b, fmt='%d')
    


env = gym.make('Pong-v4')
env.reset()

play(env, zoom=3, fps=6, callback = mycallbacks)

env.close()
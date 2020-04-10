import gym
from gym.utils.play import play
import imageio
import pdb


def mycallbacks(obs_t, obs_tp1, action, rew, done, info):
    pdb.set_trace()
    imageio.imwrite("tp1.jpg",obs_tp1)
    imageio.imwrite("t.jpg", obs_t)
    #print("action = ", action, " reward : ", rew, "done : ", done)
    #print ('tp1 : ', obs_tp1.shape, 't : ', obs_t.shape)


env = gym.make('Pong-v4')
env.reset()

play(env, zoom=3, fps=12, callback = mycallbacks)

env.close()
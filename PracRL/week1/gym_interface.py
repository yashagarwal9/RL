import numpy as np
import matplotlib.pyplot as plt
import gym
env =gym.make("MountainCar-v0")
while(1):
	plt.imshow(env.render('rgb_array'))

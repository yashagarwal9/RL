import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(3)
print(env.observation_space)
env.close()
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')		#load the environment

tf.reset_default_graph()		#clears the default graph stack

inputs1 = tf.placeholder(shape = [1, 16], dtype = tf.float32)		#Placeholder for a tensor
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))		#Uniform distribution from [0, 0.01]
Qout = tf.matmul(inputs1, W)		#([a,b,c,d]) 1x4 matrix
predict = tf.argmax(Qout, 1)		#index of max value in row(1) of matrix

nextQ = tf.placeholder(shape = [1, 4], dtype = tf.float32)		
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()
y = 0.99
e = 0.1
num_episodes = 2000
jList = []
rList = []
with tf.Session() as sess:
	sess.run(init)

	for i in range(num_episodes):
		s = env.reset()
		rAll = 0
		d = False
		j = 0
		while j < 99:
			j += 1		
			a, allQ = sess.run([predict, Qout], feed_dict = {inputs1:np.identity(16)[s:s+1]})
			if np.random.rand(1) < e:		#Choose an action by greedily (with e chance of random action) from the Q-network
				a[0] = env.action_space.sample()
			s1, r, d, _ = env.step(a[0])		#Get new state and reward from environment
			Q1 = sess.run(Qout, feed_dict = {inputs1:np.identity(16)[s1:s1+1]})		#Obtain the Q' values by feeding the new state through our network
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0, a[0]] = r + y*maxQ1
			_, W1 = sess.run([updateModel, W], feed_dict = {inputs1:np.identity(16)[s:s+1], nextQ:targetQ})
			rAll += r
			s = s1
			if d == True:
				e = 1./((i/50 + 10))
				break
		jList.append(j)
		rList.append(rAll)
		print(str(i) + ": " + str(rAll))
print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.plot(jList)
plt.show()
import gym
import numpy as np
import sys
import theano
import theano.tensor as T

import layers
from layers import *

import rl_base
from rl_base import *

env = gym.make('CartPole-v0')

"Constants"
#Number of actions
action_n = env.action_space.n

#Number of features observed
feature_n = env.observation_space.shape[0]

epochs = 100
mini_batch_n = 10#Mini batches
timestep_n = 100#Timesteps
learning_rate = 1.5

epsilon_decay_rate = -4.0/epochs
initial_epsilon = 1.0

render = False
"End Constants"

"Start network initialization"
layers = [
            FullyConnectedLayer(n_in=4, n_out=10),
            FullyConnectedLayer(n_in=10, n_out=10),
            SoftmaxLayer(n_in=10, n_out=2)
         ]

params = [param for layer in layers for param in layer.params]

x = T.vector("x")
y = T.ivector("y")
init_layer = layers[0]
init_layer.set_inpt(x, 1)

for j in xrange(1, len(layers)):
    prev_layer, layer = layers[j-1], layers[j]
    layer.set_inpt(
        prev_layer.output, 1)

"End network initialization"

"Symbolic Variable Initializations"
#For each iteration
timesteps = T.zeros(shape=timestep_n)

#For each epoch
mini_batches = T.zeros(shape=mini_batch_n)

#Global
cost = T.mean(mini_batches)
grads = T.grad(cost, params, disconnected_inputs='ignore')
updates = []

for param, grad in zip(params, grads):
    updates.append((param, param+learning_rate*grad))

"End Symbolic Variable Initializations"

"Symbolic Functions"
#To get our action a possibilities from state s
s = T.vector()
NN_eval = theano.function(
                [s], layers[-1].output,
                givens={
                    x: s
                })

train_mb = theano.function(
                [], outputs=cost, updates=updates,
                on_unused_input='warn'
                )

                
"End Symbolic Functions"


for epoch in range(epochs):
    #Decay our epsilon value(and learning rate if we want)
    epsilon = exp_decay(initial_epsilon, epsilon_decay_rate, epoch)

    #For progress
    avg_reward = 0

    for mini_batch_i in range(mini_batch_n):
        
        #Reset game environment
        state = env.reset()

        #Reset total reward for timestep t
        total_reward = 0

        #For t index in T
        for t_i in range(timestep_n):
            if render:
                env.render()

            a = NN_eval(state)
            #print "Actions: {}".format(a)

            #Get action value and action
            if epsilon_greedy(epsilon):
                #Random action
                action = env.action_space.sample()

                #Insert value for timestep(a[rng_action]
                #timesteps = T.set_subtensor(timesteps[t_i], T.ones_like(T.max(a)))
                timesteps = T.set_subtensor(timesteps[t_i], a[0][action])
                #print "Random Action: {}".format(action)

            else:
                #Policy Action

                #Insert value for timestep(max(a))
                timesteps = T.set_subtensor(timesteps[t_i], T.max(a))

                #Get actual action
                action = np.argmax(a)
                #print "Policy Action: {}".format(action)

            #Execute action and get state and reward
            state, reward, done, info = env.step(action)

            #Increment total_reward
            total_reward += reward

            #Multiply our action value by reward
            #timesteps = T.set_subtensor(timesteps[t_i], timesteps[t_i] * reward)

            #End iteration if we are done early
            if done:
                break
        
        #For progress
        avg_reward += total_reward / float(mini_batch_n)
        #print "Epoch: %i, Sample: %i, Reward: %i, Epsilon: %f" % (epoch, mini_batch_i, total_reward, epsilon)

        #Sum timestep action value vector into appropriate index in mini batch value collection
        mini_batches = T.set_subtensor(mini_batches[mini_batch_i], T.sum(timesteps)*total_reward)

        #Reset timestep action value vector to zeros
        "Note: This may not be necessary"
        #T.set_subtensor(timesteps, T.zeros(shape=timestep_n))

    #Our cost is already relative to our now complete mini batch value vector,
    #So we execute our train_mb function to evaluate new cost and update params
    #Since params are updated by grads which are updated by cost
    #params <- grads <- cost

    #cost_mb = train_mb(mini_batches)
    cost_mb = train_mb()


    #For progress
    print "Epoch: %i, Avg Reward: %i" % (epoch, avg_reward)
    print "\tActions: {}".format(a)

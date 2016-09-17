import gym
import numpy as np
import sys
import theano
import theano.tensor as T

import layers3D
from layers3D import *

import rl_base
from rl_base import *

import matplotlib.pyplot as plt#For plotting end results

env = gym.make('CartPole-v0')

"Main Parameters"
#Number of actions
action_n = env.action_space.n

#Number of features observed
feature_n = env.observation_space.shape[0]

epochs = 100
mb_n = 1#Mini batches
timestep_n = 100#Timesteps

#epsilon_decay_rate = -4.0/epochs
epsilon_decay_rate = -4.0/100
#learning_rate_decay_rate = -2.0/epochs
#epsilon_decay_rate = -0.5

initial_epsilon = 1.0
#initial_learning_rate = 1.0
learning_rate = 1.0

discount_factor = 0.95

#For display
render = False
costs = np.zeros(shape=(epochs))
avg_timesteps_global = np.zeros(shape=(epochs))

graph_cost = True
"End Main Parameters"

"Start network initialization"
layers = [
            FullyConnectedLayer(n_in=feature_n, n_out=10),
            FullyConnectedLayer(n_in=10, n_out=10),
            SoftmaxLayer(n_in=10, n_out=action_n)
         ]

params = [param for layer in layers for param in layer.params]

x = T.tensor3("x")
#y = T.tensor3("y")
init_layer = layers[0]
init_layer.set_inpt(x, mb_n, timestep_n)

for j in xrange(1, len(layers)):
    prev_layer, layer = layers[j-1], layers[j]
    layer.set_inpt(
        prev_layer.output, mb_n, timestep_n)

output = layers[-1].output
"End network initialization"

"Symbolic Variable Initializations"
#For use in computation
state = T.tensor3("state")
states = T.tensor3("states")
chosen_actions = T.tensor3("chosen_actions")
rewards = T.matrix("rewards")
grad_total_reward = T.scalar()
total_reward = T.scalar()

#Global
#Note: We have to ln(output) first because if we multiply by our one-hot 
"""
The one we think should work best currently
cost = T.mean(
            T.sum(
                T.sum(
                    T.log(output) * chosen_actions,
                axis=2),
            axis=1)
            *
            T.sum(rewards, axis=1)
        )
"""
"""
cost = T.mean(
                T.log(
                    T.sum(
                        T.sum(
                            output * chosen_actions,
                        axis=2),
                    axis=1),
                )
                *
                T.sum(rewards, axis=1)
        )
"""
"""
cost = T.mean(
            T.sum(
                T.sum(
                    T.log(output) * chosen_actions,
                axis=2),
            axis=1)
        )
"""
"""
cost = T.mean(
            T.sum(
                T.log(output) * chosen_actions,
            axis=2),
        )
"""
"""
Try multiplying our grad by total_reward
"""
grads = T.grad(cost, params)
updates = []

for param, grad in zip(params, grads):
    updates.append((param, param+learning_rate*grad*grad_total_reward))

"End Symbolic Variable Initializations"

"Symbolic Functions"
#To get our actions vector a_v from state s

NN_eval = theano.function(
                [state], outputs=output,
                givens={
                    x: state
                }
                )

#We don't output our new grad_total_reward, though it does get changed with the inputs
train_mb = theano.function(
                [states, chosen_actions, total_reward], outputs=cost, updates=updates,
                givens={
                    x: states,
                    grad_total_reward: total_reward
                }
                )
                
"End Symbolic Functions"

for epoch in range(epochs):
    #Reset these
    mb_states = np.zeros(shape=(mb_n, timestep_n, feature_n))
    mb_chosen_actions = np.zeros(shape=(mb_n, timestep_n, action_n))
    mb_rewards = np.zeros(shape=(mb_n, timestep_n))

    #Decay our epsilon value(and learning rate if we want)
    epsilon = exp_decay(initial_epsilon, epsilon_decay_rate, epoch)
    #learning_rate = exp_decay(initial_learning_rate, learning_rate_decay_rate, epoch)

    #For monitoring progress
    avg_reward = 0
    avg_timesteps = 0

    for mb_i in range(mb_n):
        
        #Reset game environment
        state = env.reset()

        #For use in measuring progress
        total_reward = 0
        total_timesteps = 0

        #For t index in T
        for t_i in range(timestep_n):
            if render:
                env.render()

            #tmp = np.zeros(shape=(mb_n, timestep_n, feature_n))
            tmp = np.zeros_like(mb_states)
            tmp[0][0] = state

            #print NN_eval(tmp).shape
            #mb_chosen_actions[:][:][1] = 1
            #print np.log(NN_eval(tmp))*mb_chosen_actions
            #Our action vector for this state
            a_v = NN_eval(tmp)[0][0]
            """
            mb_chosen_actions[0][0][0]=1
            mb_rewards[0][:]=1
            print train_mb(tmp, mb_chosen_actions, mb_rewards)
            sys.exit()
            """
            #print a_v
            #print a_v[0][0]
            #print a_v[0][1]
            #print "Actions: {}".format(a)

            #Get action value and action
            """
            EPSILON GREEDY EXPLOIT-EXPLORE
            """
            if epsilon_greedy(epsilon):
                #Random action
                action = env.action_space.sample()

            else:
                #Policy Action
                action = np.argmax(a_v)
            """
            SOFTMAX DISCRETE DISTRIBUTION EXPLOIT-EXPLORE
            action = from_discrete_dist(a_v)[0]
            """

            #Execute action and get state and reward
            state, reward, done, info = env.step(action)
            
            #Modify with discount factor if we choose to
            #reward = reward * discount_factor**(float(timestep_n)-t_i)
            #reward = reward * discount_factor**(t_i)

            #For use in measuring progress
            total_reward += reward
            total_timesteps += 1

            #Insert state, chosen action index, and reward
            mb_states[mb_i][t_i] = state
            mb_chosen_actions[mb_i][t_i][action] = 1#(Since this is a one-hot thing)
            mb_rewards[mb_i][t_i] = reward
            
            #End iteration if we are done early
            if done:
                break

        
        #For progress
        avg_timesteps += total_timesteps / float(mb_n)
        #print "Epoch: %i, Sample: %i, Reward: %i, Epsilon: %f" % (epoch, mb_i, total_reward, epsilon)

    #We execute our train_mb function to evaluate new cost and update params
    #Since params are updated by grads which are updated by cost
    #params <- grads <- cost <- inputs, chosen actions, & rewards
    #print mb_states.shape, mb_chosen_actions.shape, mb_rewards.shape
    #print mb_chosen_actions[0], total_reward
    #sys.exit()
    #total_rewards = np.ones(shape=(len(params)))*total_reward
    print total_reward
    cost_mb = train_mb(mb_states, mb_chosen_actions, total_reward)
    costs[epoch] = np.nan_to_num(cost_mb)
    avg_timesteps_global[epoch] = avg_timesteps

    #For progress
    print "Epoch: %i, Avg Timesteps: %i, Cost: %f" % (epoch, avg_timesteps, cost_mb)
    print "\tActions: {}".format(a_v)

if graph_cost:
    matplot_x = np.arange(epochs)
    plt.subplot(1, 2, 1)
    plt.plot(matplot_x, costs)
    plt.subplot(1, 2, 2)
    plt.plot(matplot_x, avg_timesteps_global)
    plt.show()

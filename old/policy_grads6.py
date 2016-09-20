"""
This one uses the baseline factor, 
and is based off of the algorithm implementation I found here

Also uses chosen actions as an index instead of one hot
http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/pg-startercode.py
"""

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

epochs = 400
mb_n = 1#Mini batches
timestep_n = 200#Timesteps

epsilon_decay_rate = -6.0/epochs
learning_rate_decay_rate = -6.0/epochs
mb_decay_rate = 2.0/epochs

initial_epsilon = 0.0
initial_learning_rate = 0.1
initial_mb_n = 1
learning_rate = initial_learning_rate#Need this defined for updates

discount_factor = 0.95

#For display
render = False
render_intermittent = False
costs = np.zeros(shape=(epochs))
avg_timesteps_global = np.zeros(shape=(epochs))
max_timesteps_global = np.zeros(shape=(epochs))
graph_cost = True
"End Main Parameters"

"Start network initialization"
layers = [
            FullyConnectedLayer(n_in=feature_n, n_out=8),
            FullyConnectedLayer(n_in=8, n_out=8),
            SoftmaxLayer(n_in=8, n_out=action_n)
         ]

params = [param for layer in layers for param in layer.params]

x = T.tensor3("x")#No actual need for a y
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
state = T.tensor3()
states = T.tensor3()
chosen_actions = T.tensor3()
advantages = T.matrix()

#Global
"""
cost = -T.mean(
            T.log(output[T.arange(output.shape[0]), T.arange(output.shape[1]), chosen_actions[T.arange(output.shape[0])] ])
                * 
            advantages
        )
"""
"""
cost = -T.mean(
            T.log(output[:, T.arange(output.shape[1]), chosen_actions[T.arange(chosen_actions.shape[0])]])
                * 
            advantages
        )
"""
cost = -T.mean(
            T.sum(T.log(output)*chosen_actions, axis=2)
                * 
            advantages
        )
grads = T.grad(cost, params)
updates = []

for param, grad in zip(params, grads):
    updates.append((param, param+learning_rate*grad))

"End Symbolic Variable Initializations"

"Symbolic Functions"
#To get our actions vector a_v from state s

NN_eval = theano.function(
                [x], outputs=output,
                )

#We don't output our new grad_total_reward, though it does get changed with the inputs
train_mb = theano.function(
                [x, chosen_actions, advantages], outputs=cost, updates=updates,
                on_unused_input='ignore'
                )
                
"End Symbolic Functions"

for epoch in range(epochs):
    mb_n = int(np.floor(exp_decay(initial_mb_n, mb_decay_rate, mb_n)))
    #Reset these
    mb_states = np.zeros(shape=(mb_n, timestep_n, feature_n))
    mb_chosen_actions = np.zeros(shape=(mb_n, timestep_n, action_n), dtype='int32')
    mb_advantages = np.zeros(shape=(mb_n, timestep_n))

    #Decay our epsilon value(and learning rate if we want)
    epsilon = exp_decay(initial_epsilon, epsilon_decay_rate, epoch)
    learning_rate = exp_decay(initial_learning_rate, learning_rate_decay_rate, epoch)

    #For monitoring progress
    #avg_reward = 0
    avg_timesteps = 0
    max_timesteps = 0

    for mb_i in range(mb_n):
        
        #Reset game environment
        state = env.reset()

        #Reset rewards
        rewards = np.zeros(shape=(timestep_n))

        #For use in measuring progress
        #total_reward = 0
        total_timesteps = 0

        #For t index in T
        for t_i in range(timestep_n):
            if render:
                env.render()
            else:
                if render_intermittent:
                    if epoch % 50 == 0 and mb_i == 0:
                        env.render()


            tmp = np.zeros(shape=(mb_n, timestep_n, feature_n))
            tmp[0][0] = state

            #Our action vector for this state
            #TODO: get this so we can input one at a time instead of zeroed 3d tensor
            a_v = NN_eval(tmp)[0][0]
            """
            print a_v
            mb_chosen_actions[0][0]=1
            mb_advantages[0][0]=-0.025
            print train_mb(tmp, mb_chosen_actions, mb_advantages)
            sys.exit()
            """

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
            reward *= discount_factor**(t_i)

            #Store reward and use it for baseline function calculation
            #Baseline is continuously updated average of rewards
            rewards[t_i] = reward
            baseline = np.mean(rewards)

            #Advantage for this timestep computed as follows:
            #Discounted reward - baseline function output for t
            advantage = reward - baseline

            #For use in measuring progress
            #total_reward += reward
            total_timesteps += 1

            #Insert state, chosen action index, and reward
            mb_states[mb_i][t_i] = state
            #mb_chosen_actions[mb_i][t_i] = int(action)#Store index
            mb_chosen_actions[mb_i][t_i][action] = 1
            mb_advantages[mb_i][t_i] = advantage
            
            #End iteration if we are done early
            if done:
                break

        #For progress
        if (total_timesteps > max_timesteps):
            max_timesteps = total_timesteps
        avg_timesteps += total_timesteps / float(mb_n)
        #print "Epoch: %i, Sample: %i, Reward: %i, Epsilon: %f" % (epoch, mb_i, total_reward, epsilon)

    #We execute our train_mb function to evaluate new cost and update params
    #Since params are updated by grads which are updated by cost
    #params <- grads <- cost <- inputs, chosen actions, & rewards
    cost_mb = train_mb(mb_states, mb_chosen_actions, mb_advantages)
    costs[epoch] = np.nan_to_num(cost_mb)
    avg_timesteps_global[epoch] = avg_timesteps
    max_timesteps_global[epoch] = max_timesteps

    #For progress
    #print "Epoch: %i, Avg Timesteps: %i, Cost: %f" % (epoch, avg_timesteps, cost_mb)
    print "Epoch: %i, Best Timesteps: %i, Cost: %f, Learning Rate: %f, Mini Batches %i" % (epoch, max_timesteps, cost_mb, learning_rate, mb_n)
    #print "\tActions: {}".format(a_v)

if graph_cost:
    matplot_x = np.arange(epochs)
    plt.subplot(1, 2, 1)
    plt.plot(matplot_x, costs)
    plt.subplot(1, 2, 2)
    #plt.plot(matplot_x, avg_timesteps_global)
    plt.plot(matplot_x, max_timesteps_global)
    plt.show()

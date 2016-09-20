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

epochs = 10000
mb_n = 10#Mini batches
timestep_n = 100#Timesteps
learning_rate = 0.1

#epsilon_decay_rate = -4.0/epochs
epsilon_decay_rate = -4.0/100.0
#epsilon_decay_rate = -0.5
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

x = T.matrix("x")
y = T.matrix("y")
init_layer = layers[0]
init_layer.set_inpt(x, mb_n*timestep_n)

for j in xrange(1, len(layers)):
    prev_layer, layer = layers[j-1], layers[j]
    layer.set_inpt(
        prev_layer.output, mb_n*timestep_n)
output = layers[-1].output

"End network initialization"

"Symbolic Variable Initializations"
#For use in computation
state = T.matrix("state")
states = T.matrix("states")
#chosen_actions = T.matrix("chosen_actions")

rewards = T.matrix("rewards")

#Global
#cost = T.mean(T.dot(T.sum(actions, axis=1), T.sum(rewards, axis=1)))
#cost = T.mean(T.dot(output, rewards))*timestep_n
#cost = T.mean(T.sum(T.log(T.dot(output, chosen_actions)), axis=1).T*rewards)*timestep_n
cost = T.mean(T.sum(T.log(output)*rewards, axis=1))*timestep_n
grads = T.grad(cost, params)
updates = []

for param, grad in zip(params, grads):
    updates.append((param, param+learning_rate*grad))

"End Symbolic Variable Initializations"

"Symbolic Functions"
#To get our actions vector a_v from state s

NN_eval = theano.function(
                [state], outputs=output,
                givens={
                    x: state
                }
                )

train_mb = theano.function(
                [states, rewards], outputs=cost, updates=updates,
                givens={
                    x: states
                }
                )
                
"End Symbolic Functions"

"Normal Variable Initializations"

"""
Do mb_n*timestep_n because it is much easier to use layers when we have
a 2d vector of input samples(matrix), rather than a 3d one(tensor 3d)

It's also nice because summing over row, then doing the mean is the
mathematical equivalent of doing the mean over all of it, then multiplying by our subsection size, aka timestep_n.
"""


"End Normal Variable Initializations"

for epoch in range(epochs):
    #Reset these
    mb_states = np.zeros(shape=(mb_n*timestep_n, feature_n))
    mb_rewards = np.zeros(shape=(mb_n*timestep_n, action_n))

    #Decay our epsilon value(and learning rate if we want)
    epsilon = exp_decay(initial_epsilon, epsilon_decay_rate, epoch)

    #For monitoring progress
    avg_reward = 0

    i=0

    for mb_i in range(mb_n):
        
        #Used at the end of each iteration, needs to be reset each time
        chosen_actions = []

        #Reset game environment
        state = env.reset()

        #For use in measuring progress
        total_reward = 0
        total_timesteps = 0

        #For t index in T
        for t_i in range(timestep_n):
            if render:
                env.render()

            #a_v = NN_eval(state.dimshuffle(0, 0, 'x'))
            #print state.reshape(1, 1, 4).shape
            #print state.reshape(1, 1, 4)
            #print NN_eval([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]], [[0.1, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]]])
            tmp = np.zeros(shape=(mb_n*timestep_n, feature_n))
            tmp[0] = state
            #print np.log(NN_eval(tmp)).shape, mb_rewards.shape, (np.log(NN_eval(tmp))*mb_rewards).shape
            a_v = NN_eval(tmp)[0]
            #a_v = NN_eval(state.reshape(1, 1, 4))
            #print a_v
            #sys.exit()
            #print "Actions: {}".format(a)

            #Get action value and action
            if epsilon_greedy(epsilon):
                #Random action

                action_i = env.action_space.sample()
                action = action_i
                #a = a_v[0][action_i]

                #print "Random Action: {}".format(action)

            else:
                #Policy Action
            
                action = np.argmax(a_v)
                #a = np.max(a_v)

                #print "Policy Action: {}".format(action)

            #Execute action and get state and reward
            state, reward, done, info = env.step(action)

            #For use in measuring progress
            total_reward += reward

            #Add new action and reward to appropriate index,
            #We get the appropriate vector representation from this rather matrix loop
            #via row_index*row_n + column_index
            mb_states[i] = state
            #mb_states[mb_i*mb_n + t_i] = state
            #mb_rewards[mb_i*mb_n + t_i] = reward

            chosen_actions.append(action)#Append action index

            i+=1
            total_timesteps+=1

            #End iteration if we are done early
            if done:
                break
        
        #Now that we have the total reward, insert it into the correct spot in our mb_rewards
        #To make a matrix with zeros everywhere but the index of the action chosen and the total reward
        for chosen_action_i, chosen_action in enumerate(chosen_actions):
            #print mb_i*mb_n+chosen_action_i, mb_rewards[mb_i*mb_n + chosen_action_i]
            #print i-total_timesteps+chosen_action_i
            mb_rewards[i-total_timesteps+ chosen_action_i][chosen_action] = total_reward

        #For progress
        avg_reward += total_reward / float(mb_n)
        #print "Epoch: %i, Sample: %i, Reward: %i, Epsilon: %f" % (epoch, mb_i, total_reward, epsilon)

    #We execute our train_mb function to evaluate new cost and update params
    #Since params are updated by grads which are updated by cost
    #params <- grads <- cost <- action & reward matrices
    #print mb_states
    #print mb_states.shape, mb_rewards.shape
    cost_mb = train_mb(mb_states, mb_rewards)
    print cost_mb


    #For progress
    print "Epoch: %i, Avg Reward: %i" % (epoch, avg_reward)
    print "\tActions: {}".format(a_v)

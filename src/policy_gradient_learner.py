
import gym
import numpy as np
import sys
import theano
import theano.tensor as T

import layers3D
from layers3D import *

import rl_base
from rl_base import *

import explore_exploit_policies
from explore_exploit_policies import *

import advantage_functions
from advantage_functions import *

import matplotlib.pyplot as plt#For plotting end results

class PolicyGradientLearner(object):
    def __init__(self, epochs, mb_n, timestep_n, initial_epsilon, initial_learning_rate, discount_factor, epsilon_decay_rate, learning_rate_decay_rate):
        "Hyper Parameters"
        self.epochs = epochs
        self.mb_n = mb_n#Mini batches
        self.timestep_n = timestep_n#Timesteps

        self.initial_epsilon = initial_epsilon
        self.initial_learning_rate = initial_learning_rate
        self.discount_factor = discount_factor

        self.epsilon_decay_rate = epsilon_decay_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        "End Hyper Parameters"

    def init_env(self, env_name):
        #Initialize our network and action environment

        "Main, Static Parameters"
        self.env = gym.make(env_name)

        #Number of actions
        self.action_n = self.env.action_space.n

        #Number of features observed
        self.feature_n = self.env.observation_space.shape[0]

        self.explore_exploit_policy = epsilon_greedy()
        self.advantage_function = updated_mean()

        self.learning_rate = self.initial_learning_rate#Need this defined for updates

        #For display
        self.render = False
        self.render_intermittent = False
        self.costs = np.zeros(shape=(self.epochs))
        self.avg_timesteps_global = np.zeros(shape=(self.epochs))
        self.max_timesteps_global = np.zeros(shape=(self.epochs))
        self.graph_cost = True
        "End Main, Static Parameters"

        "Start network initialization"
        layers = [
                    FullyConnectedLayer(n_in=self.feature_n, n_out=8),
                    FullyConnectedLayer(n_in=8, n_out=8),
                    SoftmaxLayer(n_in=8, n_out=self.action_n)
                 ]

        params = [param for layer in layers for param in layer.params]

        x = T.tensor3("x")#No actual need for a y
        init_layer = layers[0]
        init_layer.set_inpt(x, self.mb_n, self.timestep_n)

        for j in xrange(1, len(layers)):
            prev_layer, layer = layers[j-1], layers[j]
            layer.set_inpt(
                prev_layer.output, self.mb_n, self.timestep_n)

        output = layers[-1].output
        "End network initialization"

        "Symbolic Variable Initializations"
        #For use in computation
        state = T.tensor3()
        states = T.tensor3()
        chosen_actions = T.tensor3()
        advantages = T.matrix()

        #Global
        cost = -T.mean(
                    T.sum(T.log(output)*chosen_actions, axis=2)
                        * 
                    advantages
                )
        grads = T.grad(cost, params)
        updates = []

        for param, grad in zip(params, grads):
            updates.append((param, param+self.learning_rate*grad))

        "End Symbolic Variable Initializations"

        "Symbolic Functions"
        #To get our actions vector a_v from state s

        self.network_eval = theano.function(
                        [x], outputs=output, allow_input_downcast=True
                        )

        #We don't output our new grad_total_reward, though it does get changed with the inputs
        self.train_mb = theano.function(
                        [x, chosen_actions, advantages], outputs=cost, updates=updates, allow_input_downcast=True
                        )
                        
        "End Symbolic Functions"

    def train_env(self, config_i, config_n, run_i, run_n):
        for epoch in range(self.epochs):

            #self.mb_n = int(np.floor(exp_decay(initial_self.mb_n, mb_decay_rate, epoch)))
            #print np.floor(exp_decay(initial_self.mb_n, mb_decay_rate, epoch))

            #Reset these
            mb_states = np.zeros(shape=(self.mb_n, self.timestep_n, self.feature_n))
            mb_chosen_actions = np.zeros(shape=(self.mb_n, self.timestep_n, self.action_n), dtype='int32')
            mb_advantages = np.zeros(shape=(self.mb_n, self.timestep_n))

            #Decay our epsilon value(and learning rate if we want)
            epsilon = exp_decay(self.initial_epsilon, self.epsilon_decay_rate, epoch)
            self.learning_rate = exp_decay(self.initial_learning_rate, self.learning_rate_decay_rate, epoch)

            #For monitoring progress
            #avg_reward = 0
            avg_timesteps = 0
            max_timesteps = 0

            for mb_i in range(self.mb_n):
                
                #Reset game environment
                state = self.env.reset()

                #Reset rewards
                rewards = np.zeros(shape=(self.timestep_n))

                #For use in measuring progress
                #total_reward = 0
                total_timesteps = 0

                #For t index in T
                for t_i in range(self.timestep_n):
                    if self.render:
                        self.env.render()
                    else:
                        if self.render_intermittent:
                            if epoch % 50 == 0 and mb_i == 0:
                                self.env.render()

                    tmp = np.zeros(shape=(self.mb_n, self.timestep_n, self.feature_n))
                    tmp[0][0] = state

                    #Our action vector for this state
                    #TODO: get this so we can input one at a time instead of zeroed 3d tensor
                    a_v = self.network_eval(tmp)[0][0]

                    #Get action value and action
                    action = self.explore_exploit_policy.get_action(epsilon, a_v)

                    #Execute action and get state and reward
                    state, reward, done, info = self.env.step(action)
                    
                    #Modify with discount factor if we choose to
                    reward *= self.discount_factor**(t_i)

                    #Store reward and use it for baseline function calculation
                    rewards[t_i] = reward

                    #Get advantage using our advantage function class
                    advantage = self.advantage_function.get_advantage(rewards, reward)

                    #For use in measuring progress
                    #total_reward += reward
                    total_timesteps += 1

                    #Insert state, chosen action index, and reward
                    mb_states[mb_i][t_i] = state
                    mb_chosen_actions[mb_i][t_i][action] = 1
                    mb_advantages[mb_i][t_i] = advantage
                    
                    #End iteration if we are done early
                    if done:
                        break

                #For progress
                if (total_timesteps > max_timesteps):
                    max_timesteps = total_timesteps
                avg_timesteps += total_timesteps / float(self.mb_n)
                #print "Epoch: %i, Sample: %i, Reward: %i, Epsilon: %f" % (epoch, mb_i, total_reward, epsilon)

            #We execute our self.train_mb function to evaluate new cost and update params
            #Since params are updated by grads which are updated by cost
            #params <- grads <- cost <- inputs, chosen actions, & rewards
            cost_mb = self.train_mb(mb_states, mb_chosen_actions, mb_advantages)
            self.costs[epoch] = np.nan_to_num(cost_mb)
            self.avg_timesteps_global[epoch] = avg_timesteps
            self.max_timesteps_global[epoch] = max_timesteps

            #For progress
            #print "Epoch: %i, Avg Timesteps: %i, Cost: %f" % (epoch, avg_timesteps, cost_mb)
            sys.stdout.write("\rConfig %i/%i, Run %i/%i, Epoch: %i/%i, Avg Timesteps: %i/%i\t" % (config_i+1, config_n, run_i+1, run_n, epoch+1, self.epochs, avg_timesteps, self.timestep_n))
            sys.stdout.flush()
            """
            print "\tInitial Learning Rate: %f" % (self.initial_learning_rate)
            print "\tLearning Rate Decay Rate: %f" % (self.learning_rate_decay_rate)
            print "\tMini-Batch Size: %i" % (self.mb_n)
            print "\tDiscount Factor: %f" % (self.discount_factor)
            """
            #print "\tActions: {}".format(a_v)

        """
        if self.graph_cost:
            matplot_x = np.arange(self.epochs)
            plt.subplot(1, 2, 1)
            plt.plot(matplot_x, self.costs)
            plt.subplot(1, 2, 2)
            #plt.plot(matplot_x, self.avg_timesteps_global)
            plt.plot(matplot_x, self.max_timesteps_global)
            plt.show()
        """

        #Return a 2d array so that we have each element represent the values at each timestep
        #Holy shit this is the best np method ever
        return np.column_stack((self.costs, self.avg_timesteps_global, self.max_timesteps_global))

import gym
import numpy as np
import sys
from tqdm import tqdm

import keras.backend as K
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

import explore_exploit_policies
from explore_exploit_policies import *

import advantage_functions
from advantage_functions import *

import matplotlib.pyplot as plt

class PolicyGradientModel(object):
    def __init__(self, env_name, epochs=100, mb_n=20, timesteps=200, epsilon=0.5, learning_rate=1.0, discount_factor=1.0):

        self.env_name = env_name
        self.epochs = epochs
        self.mb_n = mb_n
        self.timesteps = timesteps

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        #So that these will decay to .0001 by the final epoch
        self.epsilon_decay_rate = -6.0/self.epochs
        self.learning_rate_decay_rate = -6.0/self.epochs

        #Initialize our environment 
        print("Initializing Environment...")
        self.env = gym.make(env_name)

        #State and action shapes give us number of input and output features of the model
        self.input_n, self.output_n = self.env.observation_space.shape[0], self.env.action_space.n

        #TODO: Change these for hard functions?
        self.explore_exploit_policy = epsilon_greedy()
        self.advantage_function = updated_mean()

        #For display
        self.render = False
        self.render_intermittent = False

        #For stats
        self.costs = np.zeros((self.epochs, 2))
        self.avg_rewards = np.zeros((self.epochs))
        self.avg_timesteps = np.zeros((self.epochs))

        #For training
        self.sample_n = self.mb_n*self.timesteps #Number of samples in each batch
        self.states = np.zeros((self.sample_n, self.input_n))#Inputs to NN, states
        self.predictions = np.zeros((self.sample_n, self.output_n))#Actual outputs of NN (softmax)
        self.actions = np.zeros((self.sample_n, self.output_n))#Chosen actions based on explore-exploit policy (one-hot)
        self.advantages = np.zeros((self.sample_n))#Reward of each chosen action

        self.rewards = np.zeros((self.timesteps))

        #Define and Initialize our model
        print("Initializing Model...")

        #loss = "categorical_crossentropy"
        #optimizer = Adam(1e-4)
        optimizer = SGD(lr=self.learning_rate, decay=-self.learning_rate_decay_rate)

        self.model = Sequential()
        self.model.add(Dense(8, activation="sigmoid", input_shape=(self.input_n,)))
        self.model.add(Dense(8, activation="sigmoid"))
        self.model.add(Dense(8, activation="sigmoid"))
        self.model.add(Dense(8, activation="sigmoid"))
        self.model.add(Dense(8, activation="sigmoid"))
        self.model.add(Dense(8, activation="sigmoid"))
        self.model.add(Dense(self.output_n, activation="softmax"))
        self.model.compile(loss=self.pgrad_loss, optimizer=optimizer, metrics=["accuracy"])

    def pgrad_loss(self, predictions, actions):
        return K.mean(K.sum(K.log(predictions)*actions, axis=1) * self.advantages)

    def train(self):

        print("Training Model on Environment...")
        for epoch in range(self.epochs):
            #Decay our epsilon value and learning rate by their decay rates
            self.epsilon *= np.exp(self.epsilon_decay_rate)
            #self.learning_rate *= np.exp(learning_rate_decay_rate)

            for mb in range(self.mb_n):
                #TODO: Make stepping parallel because otherwise we literally predict at every step
                
                #Reset game environment
                state = self.env.reset()

                for timestep in range(self.timesteps):
                    if self.render:
                        self.env.render()
                    else:
                        if self.render_intermittent:
                            if epoch % 100 == 0 and mb == 0:
                                self.env.render()

                    #Get prediction vector from model and choose action index using our explore exploit policy
                    prediction = self.model.predict(np.array([state]))
                    action = self.explore_exploit_policy.get_action(self.epsilon, prediction)

                    #Execute action to get state and reward
                    state, reward, done, info = self.env.step(action)
                    
                    #Discount reward
                    self.rewards[timestep] = reward*self.discount_factor**(timestep)

                    #Get advantage using our advantage function 
                    advantage = self.advantage_function.get_advantage(self.rewards[:timestep+1], self.rewards[timestep])

                    #Insert values for this timestep at proper sample index
                    sample_i = mb*self.mb_n+timestep
                    self.states[sample_i] = state
                    self.actions[sample_i][action] = 1.0
                    self.advantages[sample_i] = advantage
                    
                    #Increment these sums so they can be averaged after we're done with this epoch
                    self.avg_rewards[epoch] += self.rewards[timestep]
                    self.avg_timesteps[epoch] += 1

                    #End iteration if we are done early / achieved a goal
                    if done:
                        break

            #Train the network for this one mini batch
            cost = self.model.train_on_batch(self.states, self.actions)

            #Save performance stats for this epoch
            self.costs[epoch] = np.nan_to_num(cost)
            self.avg_rewards[epoch] /= self.mb_n#Compute average from the sum
            self.avg_timesteps[epoch] /= self.mb_n#Compute average from the sum

            sys.stdout.write("\rEpoch: {}, Costs: {}, Avg. Rewards: {}, Avg Timesteps: {}, Epsilon: {}, Learning Rate: {}".format(epoch+1, self.costs[epoch], self.avg_rewards[epoch], self.avg_timesteps[epoch], self.epsilon, self.learning_rate))
            sys.stdout.flush()

        plt.subplot(1, 2, 1)
        plt.plot(self.avg_rewards)
        plt.subplot(1, 2, 2)
        plt.plot(self.avg_timesteps)
        plt.show()


            

#pgm = PolicyGradientModel("Acrobot-v1", epochs=100, mb_n=20, timesteps=200, epsilon=1.0, learning_rate=3.0)
pgm = PolicyGradientModel("CartPole-v0", epochs=4000, mb_n=10, timesteps=200, epsilon=1.0, learning_rate=1.0, discount_factor=0.95)
pgm.train()
print("")

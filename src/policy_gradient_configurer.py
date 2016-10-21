
import policy_gradient_learner
from policy_gradient_learner import *

import numpy as np

class Configurer(object):

    def __init__(self, epochs, timestep_n):
        self.epochs = epochs
        self.timestep_n = timestep_n

    def run_config(self, config_i, config_n, run_count, mb_n, initial_epsilon, initial_learning_rate, epsilon_decay_rate, learning_rate_decay_rate, discount_factor):
        #Need this
        mb_n = int(mb_n)
        if mb_n == 0:
            mb_n = 1

        #So these are optimized in a way that they still work on different E value (/epochs)
        #Negative value so we can still use our hyperparameter class and not go outside of range necessary
        epsilon_decay_rate = -epsilon_decay_rate/self.epochs
        learning_rate_decay_rate = -float(learning_rate_decay_rate)/self.epochs

        cartpole_agent = PolicyGradientLearner(self.epochs, mb_n, self.timestep_n, initial_epsilon, initial_learning_rate, discount_factor, epsilon_decay_rate, learning_rate_decay_rate)

        #For results, becomes 3d array of shape runs, epochs, values
        results = []
        for run_i in range(run_count):
            #Reset environment
            cartpole_agent.init_env('CartPole-v0')

            #Gain more results for mean
            results.append(cartpole_agent.train_env(config_i, config_n, run_i, run_count))

        #Now we have 2d array of shape epochs, values
        average_results = np.mean(results, axis=0)

        #Ok this ones a bit complicated
        #We need to get a list like 1, 2, 3 if the number of values we get from this is 3, hence the
        #i+1 for i in range(len(average_results[0])-1)

        #Then, we do hsplit to split our matrix on the columns(since we have 0, 1, 2, all the column indices)
        #and thus get our indepentent average values for each one
        average_values = np.hsplit(average_results, np.array([i+1 for i in range(len(average_results[0])-1)]))#so many brackets asdfahlkasdf))Fasdf0))))

        #So we can transpose the column vector back to a row one for each of these
        average_values = [average_value.flatten() for average_value in average_values]

        #Yay, assign values
        average_costs, average_avg_timesteps, average_max_timesteps = average_values

        return average_avg_timesteps

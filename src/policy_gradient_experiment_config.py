
import policy_gradient_learner
from policy_gradient_learner import *

import numpy as np

import matplotlib.pyplot as plt#For plotting end results

run_count = 3

epochs = 400
mb_n = 80#Mini batches
timestep_n = 200#Timesteps

initial_epsilon = 0.0
initial_learning_rate = 2.8
discount_factor = 0.9

epsilon_decay_rate = -6.0/epochs
learning_rate_decay_rate = -4.0/epochs

cartpole_agent = PolicyGradientLearner(epochs, mb_n, timestep_n, initial_epsilon, initial_learning_rate, discount_factor, epsilon_decay_rate, learning_rate_decay_rate)

#For results, becomes 3d array of shape runs, epochs, values
results = []
for r in range(run_count):
    #Reset environment
    cartpole_agent.init_env('CartPole-v0')

    #Gain more results for mean
    results.append(cartpole_agent.train_env(0, 1, 0, 1))

#Now we have 2d array of shape epochs, values
average_results = np.mean(results, axis=0)

#Ok this ones a bit complicated
#We get a list like 0, 1, 2 if the number of values we get from this is 3, hence the
#i for i in range(len(average_results[0]))

#Then, we do hsplit to split our matrix on the columns(since we have 0, 1, 2, all the column indices)
#and thus get our indepentent average values for each one
#print average_results.shape
average_values = np.hsplit(average_results, np.array([i for i in range(len(average_results[0]))]))

#print [average_value.shape for average_value in average_values]
#So we can transpose the column vector back to a row one for each of these
average_values = [average_value.flatten() for average_value in average_values]

#print average_values
#print [average_value.shape for average_value in average_values]
#Yay, assign values
average_costs, average_avg_timesteps, average_max_timesteps = average_values[1:]


matplot_x = np.arange(epochs)

plt.subplot(1, 3, 1)
plt.plot(matplot_x, average_costs)

plt.subplot(1, 3, 2)
plt.plot(matplot_x, average_avg_timesteps)

plt.subplot(1, 3, 3)
plt.plot(matplot_x, average_max_timesteps)

plt.show()

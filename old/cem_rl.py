import gym
import numpy as np
import sys

env = gym.make('CartPole-v0')

#Number of actions
action_n = env.action_space.n

#Number of features observed
feature_n = env.observation_space.shape[0]
 
iterations = 1000
theta_n = 100
timesteps = 1000

avg_solved_perc = 97.5
avg_solved_threshold = (avg_solved_perc/100*timesteps)

initial_elite_p = 0.05
#make this 0 for no change in # of elites
elite_rate = 0
#elite_rate = -.33

render = False

#Initial Normal Distribution Mean and Var
#We want feature/observation number * action number for our mean & variance
#So in this case, it's 4x2
mean = np.zeros(shape=feature_n*action_n+1*action_n)
variance = np.ones(shape=feature_n*action_n+1*action_n)

def get_action(observation, theta):
    #Using the weights and biases stored in our theta, 
    #compute the best action for each observation using the following

    #Get the weights (In this instance of shape 4, 2)
    weights = theta[:feature_n*action_n].reshape(feature_n, action_n)

    #Get the biases (In this instance of shape 1, 2 because diag()
    biases = theta[feature_n*action_n:].reshape(1, action_n)

    #SOMEHOW THIS RESULTS IN (1, 2) MATRIX IDFK
    z = np.dot(observation, weights) + biases

    #Put it through activation function exp
    a = np.exp(z)
    
    #Reshape
    a = a.reshape(action_n)

    #Return the best action according to which has higher value
    return np.argmax(a)

#Iteration Loop
for i in range(iterations):
    
    #Generate multivariate normal distribution from our mean and variance
    #Use it to get our samples for this iteration
    thetas = np.random.multivariate_normal(mean, np.diag(variance), theta_n)

    #For the results of each theta
    rewards = np.zeros(shape=theta_n)

    for theta_i, theta in enumerate(thetas):
        observation = env.reset()
        done = False

        #For our later sorting by reward
        total_reward = 0

        #Time step loop
        for t in range(timesteps):
            if render:
                env.render()#Disable for fast training

            #Get our action using our current timestep observation and our theta weights & biases
            action = get_action(observation, theta)
            #action = env.action_space.sample()

            if done:
                break

            #Execute action, get reward
            observation, reward, done, info = env.step(action)

            total_reward+=reward
        #print "Iteration: %i, Theta: %i, Total Reward: %i" % (i, theta_i, total_reward)
        sys.stdout.write("\rIteration: %i, Theta: %i, Total Reward: %i" % (i, theta_i, total_reward))
        sys.stdout.flush()

        #Now that we've finished running that theta,
        #Record total reward
        rewards[theta_i] = total_reward

    #Now that we've evaluated each theta,
    #Sort descending by reward
    avg_reward = np.mean(rewards)
    rewards = rewards.argsort()[::-1]

    #Get our new number of elites
    elite_p = initial_elite_p * np.exp(elite_rate*i)
    elite_n = int(theta_n * elite_p)

    #Get our top elites
    elites = rewards[:elite_n]

    #Set our mean and variance to correct dims before updating
    mean = np.zeros(shape=feature_n*action_n+1*action_n)
    variance = np.zeros(shape=feature_n*action_n+1*action_n)

    #Update mean and variance
    for e in range(elite_n):
        mean += thetas[elites[e]]
        variance += thetas[elites[e]]**2

    mean = mean / elite_n
    variance = variance / elite_n - mean**2

    if avg_reward >= avg_solved_threshold:
        #We've solved it, start our infinite trials
        break

while True:
    #Infinite run of solved thetas

    for theta_i, theta in enumerate(thetas):
        observation = env.reset()
        done = False

        #Time step loop, want to run until it dies
        while True:
            env.render()#Disable for fast training

            #Get our action using our current timestep observation and our theta weights & biases
            action = get_action(observation, theta)
            #action = env.action_space.sample()

            if done:
                #print "Iteration: %i, Timestep: %i" % (i, t)
                #sys.stdout.write("\rIteration: %i, Timestep: %i" % (i, t))
                #sys.stdout.flush()
                break

            #Execute action, get reward
            observation, reward, done, info = env.step(action)

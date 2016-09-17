import gym

import sys
import numpy as np

env = gym.make('CartPole-v0')

#Episodes == Iterations
iterations = 10000
timesteps = 100
action_n = 2
feature_n = 4

epsilon_decay_rate = -0.0006
alpha_decay_rate = -0.0006

initial_epsilon = 1.0
initial_alpha = 1.0

discount_factor = 0.95
bias = 1.0

policy = {}#Basically q_vals
weights = np.zeros(shape=feature_n*2)


def exp_decay(initial, rate, iteration):
    #Do our k*e^(r*t) exponential decay
    return initial*np.exp(rate*iteration)

def epsilon_greedy(epsilon):
    #Return True if exploring, False if exploiting
    r = np.random.rand(1)[0]
    if r < epsilon:
        return True
    else:
        return False

def rand_action(action_n):
    #Pick a random action
    #If we have 2 actions, then it will return 0 or 1
    return np.random.randint(action_n)

def policy_action(state, policy, action_n):
    #Do our max(a) Q(s, a) using our accumulated policy / q_vals
    """
    state = tuple(state)
    if state in policy:
        q_vals = policy[state]
        return int(np.amax(q_vals))
    else:
        policy[state] = np.zeros(shape=action_n, dtype=np.float16)
        return rand_action(action_n)
    """
    q_vals = []
    for action in range(action_n):
        if action == 0:
            q_action = -1
        else:
            q_action = 1
        q_vals.append(np.sum([weight*q_action*(feature**(feature_n+1-feature_i)) for feature_i, (weight, feature) in enumerate(zip(weights, state))]) + bias)
    return np.argmax(q_vals)
    


for i in range(iterations):
    #Reset our environment
    state = env.reset()
    action = 0

    #Decay our epsilon and alpha
    epsilon = exp_decay(initial_epsilon, epsilon_decay_rate, i)
    alpha = exp_decay(initial_alpha, alpha_decay_rate, i)

    for t in range(timesteps):
        #Update our environment GUI 
        #env.render()

        #For updating our policy
        last_state = tuple(state)#Have to tuple so it can be used as a key
        last_action = action

        #Get our next action type using epsilon greedy approach
        if epsilon_greedy(epsilon):
            #Random action
            #print "Random action"
            action = rand_action(action_n)
        else:
            #Policy action
            #print "Policy action"
            action = policy_action(state, policy, action_n)

        #Calculuate value of current state before performing action
        #state == our features f1, f2, f3, f4 (<3 u OpenAI)
        #Set each feature's power = n + 1 - i, so that we have max degree of # of features down to 1 when i = n
        if action == 0:
            q_action = -1
        else:
            q_action = 1


        q_val_tmp = []
        weight_i = 0
        for feature_i, feature in enumerate(state):
            q_val_tmp.append(weights[weight_i]*q_action*(feature**(feature_n+1-feature_i)))
            q_val_tmp.append(weights[weight_i+1]*(feature**(feature_n+1-feature_i)))
            weight_i += 2

        q_val = np.sum(q_val_tmp) + bias
        #q_val = np.sum([weight*q_action*(feature**(feature_n+1-feature_i)) for feature_i, (weight, feature) in enumerate(zip(weights, state))]) + bias

        #Update our policy for the last state with the q value we got after executing the action
        """
        if last_state not in policy:
            policy[last_state] = np.zeros(shape=action_n, dtype=np.float16)
        #print "before: {}".format(policy[last_state])
        policy[last_state][last_action] = q_val
        #print state, weights
        #print q_val
        #print "after: {}".format(policy[last_state])
        """

        #Execute action
        state, reward, done, info = env.step(action)

        #Get new q_val for our new state to calculate error with
        if done:
            #Our action Q(s', a') = 0
            new_q_val = 0.0
        else:
            #Our action Q(s', a') obtained from our policy
            new_q_val = policy_action(state, policy, action_n)

        #print new_q_val, q_val
        #Calculate error
        #error = (reward + discount_factor * new_q_val) - q_val#Basically y - a, actual - desired
        error = reward - q_val
        
        #Update weights
        weight_i = 0
        weights_tmp = []
        for feature in enumerate(state):
            weights_tmp.append(weights[weight_i] + alpha * error * feature)
            weights_tmp.append(weights[weight_i+1] + alpha * error * feature)

            
        weights = [weight + alpha * error * feature for weight, feature in zip(weights, state)]

        if done:
            #print("Iteration %i finished after %i timesteps" % (i, t))
            sys.stdout.write("\rIteration: %i, Timestep: %i, Epsilon: %f, Alpha: %f" % (i, t, epsilon, alpha))
            sys.stdout.flush()
            '''
            if i == 1:
                sys.exit()
            '''
            break

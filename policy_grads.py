import gym
import numpy as np
import sys
import theano
import theano.tensor as T
import layers
from layers import FullyConnectedLayers, SoftmaxLayers
from theano.tensor.nnet import softmax
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

env = gym.make('CartPole-v0')

#Number of actions
action_n = env.action_space.n

#Number of features observed
feature_n = env.observation_space.shape[0]

epochs = 100
mini_batch_size = 10
timesteps = 100

avg_solved_perc = 97.5
avg_solved_threshold = (avg_solved_perc/100*timesteps)

render = False

def get_action(observation):

    #test_x, test_y = test_data

    i = T.lscalar() # mini-batch index

    self.test_mb_predictions = theano.function(
        [i], self.layers[-1].y_out,
        givens={
            self.x: observation
        }, on_unused_input='warn')

    return self.test_mb_predictions(0)



#Initialize network
layers = [
            FullyConnectedLayer(n_in=4, n_out=10),
            FullyConnectedLayer(n_in=10, n_out=10),
            SoftmaxLayer(n_in=10, n_out=2)
         ]

params = [param for layer in layers for param in layer.params]
iterations = mini_batch_size

x = T.matrix("x")
y = T.ivector("y")
init_layer = layers[0]
init_layer.set_inpt(x, mini_batch_size)

for j in xrange(1, len(layers)):
    prev_layer, layer = layers[j-1], layers[j]
    layer.set_inpt(
        prev_layer.output, mini_batch_size)

output = layers[-1].output
cost = T.log(output)
grads = theano.shared([np.zeros(shape=param.get_value().shape, dtype=theano.config.floatX) for param in params for m in range(mini_batch_size)])
rewards = theano.shared(np.zeros(shape=(1, mini_batch_size)))
updates = []

#Epoch loop
for epoch in range(epochs):
    #Iteration / Sample Loop
    for i in range(iterations):
        total_reward = 0

        observation = env.reset()
        #Time step loop
        for t in range(timesteps):
            if render:
                env.render()#Disable for fast training

            #grad += T.grad(cost, params)
            updates.append(grads[i], grads[i] + T.grad(cost, params))

            #action = get_action(observation, theta)
            action = get_action(observation)

            #Execute action, get reward
            observation, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                break

        updates.append(rewards[i], total_reward)
        #grad *= total_reward
        #grads[i] = grad
    updates.append(grads, grads*total_reward)
    #grads = T.mean(grads, axis=0)
    updates.append(params, params + learning_rate*grads)



"""
while True:
    #Infinite run of solved thetas

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

"""

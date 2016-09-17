import gym
import numpy as np
import sys
import theano
import theano.tensor as T
import layers

from layers import FullyConnectedLayer, SoftmaxLayer

env = gym.make('CartPole-v0')

#Number of actions
action_n = env.action_space.n

#Number of features observed
feature_n = env.observation_space.shape[0]

epochs = 100
mini_batch_size = 10
timesteps = 100
learning_rate = 1.0

epsilon_decay_rate = -0.04
initial_epsilon = 1.0
#avg_solved_perc = 97.5
#avg_solved_threshold = (avg_solved_perc/100*timesteps)

render = False

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

x = T.vector("x")
y = T.ivector("y")
init_layer = layers[0]
init_layer.set_inpt(x, 1)

for j in xrange(1, len(layers)):
    prev_layer, layer = layers[j-1], layers[j]
    layer.set_inpt(
        prev_layer.output, 1)

cost = T.argmax(T.log(layers[-1].output))
R = 0

#iter_grads = [theano.shared([np.zeros(shape=param.get_value().shape, dtype=theano.config.floatX) for param in params])]
#grads = [theano.shared([np.zeros(shape=param.get_value().shape, dtype=theano.config.floatX) for param in params])]
grads = T.grad(cost, params)
iter_grads = [T.zeros_like(grad) for grad in grads]

t_updates = []
iter_updates = []
mb_updates = []

#t_updates.append((iter_grads, iter_grads+T.grad(cost, params)))

#iter_updates.append((iter_grads, T.dot(T.dot(iter_grads, R), 1/mini_batch_size)))
#iter_updates.append((grads, grads+iter_grads))

#mb_updates.append((params, params+learning_rate*grads))
for param, grad in zip(params, grads):
    mb_updates.append((param, param+learning_rate*grad))#Update our params as we were

#To execute our updates when necessary
exec_t_updates = theano.function([], None, updates=t_updates)
exec_iter_updates = theano.function([], None, updates=iter_updates)
#exec_mb_updates = theano.function([], None, updates=mb_updates)
"""
mb = T.iscalar()
train_mb = theano.function(
        [], cost, updates=mb_updates)
"""

#To get our action a possibilities from state s
s = T.vector()
NN_output = theano.function(
                [s], layers[-1].output,
                givens={
                    x: s
                })


for e in range(epochs):
    #grads = T.set_subtensor(grads, T.zeros_like(grads))
    grads = grads * 0
    epsilon = exp_decay(initial_epsilon, epsilon_decay_rate, e)

    for mb in range(mini_batch_size):
        
        s = env.reset()
        R = 0
        #iter_grads = T.set_subtensor(iter_grads, T.zeros_like(iter_grads))
        iter_grads = grads * 0

        for t in range(timesteps):
            if render:
                env.render()

            if epsilon_greedy(epsilon):
                #Random action
                action = env.action_space.sample()
                tmp = T.scalar("tmp")
                max_action = T.ones_like(tmp)
            else:
                #Policy Action
                a = NN_output(s)
                action = np.argmax(a, axis=1)[0]
                max_action = T.max(a)

            #exec_t_update()
            iter_grads = iter_grads + T.grad(max_action, params)
            
            s, r, done, info = env.step(action)

            R += r

            if done:
                break
        #exec_iter_update()
        iter_grads = [iter_grad * R / mini_batch_size for iter_grad in iter_grads]
        grads += iter_grads

    print "Epoch: %i, Reward: %i, Epsilon: %f" % (e, R, epsilon)
    #exec_mb_updates()
    #cost_asdf = train_mb()
    #print "Updating params..."
    for param, grad in zip(params, grads):
        param = param + learning_rate * grad

#Description

This repository consists of my policy gradient reinforcement learning code, currently implemented for the classic control environment, [CartPole](https://gym.openai.com/envs/CartPole-v0) however I have done my best to make it as modular as possible to allow implementation into any of the other OpenAI environments, as well as other problems entirely. I used the extremely helpful resources provided by [OpenAI's gym](https://gym.openai.com/envs) to get the fitness, environmental features, etc. for now.

Using this code, I was able to complete the cartpole challenge, as presented on the OpenAI environment description(however I will be applying it to others soon). Here's how it works:

The general idea behind using policy gradients (as far as I have gathered), is to use (stochastic/non-stochastic) gradient descent to update parameters of any number of things depending on the environment. I used a neural network, originally with the 4 inputs, a 4-hidden-neuron fully connected sigmoid layer, and a 2 output softmax layer for the actions. Note: you can also use the mean and variance of distributions for the weights, however I haven't done that yet, and was much more excited about the combination of neural network machine learning and fitness reinforcement machine learning.

I won't go into the detailed intricacies here, however you can find a working implementation complete with the following features in this repository at the time of this writing:
  1. 3D input of neural network in order to use mini batches / stochastic gradient descent
  2. Exponential decay/growth of epsilon, learning rate, and mini batch size(last one in progress)
  3. Epsilon-greedy exploration/exploit approach option
  4. Softmax discrete distribution drawing exploration/exploit approach option
  5. Repeatedly updated mean of rewards up to the current timestep for baseline algorithm implementation (See resources if you don't know what this is)
  6. Matplotlib graphed output of results
  7. Reward Discount factor
  8. Added [CHO](https://github.com/DarkElement75/cho) and cleaned up a bit, automatically optimizes training hyper parameters on environment.

TODO:
  1. Bayesian Hyper Parameter Optimization as an upgrade to CHO will be reflected in here as well as several other repos(Currently working on this)
  2. Support for other types of action spaces, seeing as cartpole is better done with the Cross Entropy Method I will likely adjust Policy Gradients to modify the mean and variance of multiple Gaussian Distributions to draw parameters from them, instead for instance.
  3. Various other improvements and upgrades as I see them, including any cleverness I figure out for CHO or Policy Gradients

I also included my Cross-Entropy Method for Black Box optimization in here, in case you'd like that too.

###If you want good resources for understanding policy gradient reinforcement learning better, look here:

[OpenAI's tutorial](https://gym.openai.com/docs/rl#policy-gradients) (The entirety of this site is awesome for starting out)
[Some slides on it](http://home.deib.polimi.it/restelli/MyWebSite/pdf/rl7.pdf)
[More slides](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/lecture12.pdf)
[Good extra, neat examples of the algorithm along with a baseline algorithm](http://www.scholarpedia.org/article/Policy_gradient_methods)

###(Wait, isn't it gradient ascent instead of descent? hmm...)
